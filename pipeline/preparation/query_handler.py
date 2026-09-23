#query_handler.py


import os
import glob
import numpy as np
import torch
import faiss

class QueryHandler:
    """Enhanced query handler optimized for MAP evaluation."""

    def __init__(self, feature_extractor, feature_dir, use_cosine=True):
        self.feature_extractor = feature_extractor
        self.feature_dir = feature_dir
        self.use_cosine = use_cosine

        # Validate feature directory
        if not os.path.isdir(feature_dir):
            raise ValueError(f"Feature directory not found: {feature_dir}")

        # Load features with error handling
        try:
            self._load_features()
            self._build_index()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QueryHandler: {e}")

    def _load_features(self):
        """Load features with proper error handling."""
        feature_paths = sorted(glob.glob(os.path.join(self.feature_dir, '*.pt')))
        if not feature_paths:
            raise ValueError(f"No .pt files found in {self.feature_dir}")

        self.fns = []
        features = []
        failed_files = []

        for p in feature_paths:
            try:
                # Load feature and ensure proper shape
                fn = os.path.splitext(os.path.basename(p))[0]  # This removes .pt extension
                fv = torch.load(p, map_location='cpu')

                if isinstance(fv, torch.Tensor):
                    arr = fv.detach().cpu().numpy().flatten()
                else:
                    arr = np.array(fv).flatten()

                # Skip invalid features
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)) or arr.size == 0:
                    print(f"Warning: Invalid feature in {fn} - skipping")
                    failed_files.append(fn)
                    continue

                self.fns.append(fn)  # Store filename without extension
                features.append(arr)

            except Exception as e:
                print(f"Error loading {p}: {e}")
                failed_files.append(os.path.splitext(os.path.basename(p))[0])
                continue

        if not features:
            raise ValueError("No valid features loaded")

        if failed_files:
            print(f"Failed to load {len(failed_files)} feature files")

        self.features = np.vstack(features).astype('float32')

        # Normalize features if using cosine similarity
        if self.use_cosine:
            norms = np.linalg.norm(self.features, axis=1, keepdims=True)
            # Handle zero vectors
            zero_mask = (norms.flatten() == 0)
            if np.any(zero_mask):
                print(f"Warning: {np.sum(zero_mask)} zero-norm features found")
                norms[zero_mask] = 1.0
            self.features = self.features / norms

        print(f"Loaded {len(self.fns)} valid features")

    def _build_index(self):
        """Build FAISS index optimized for the similarity metric."""
        try:
            if self.use_cosine:
                # Use Inner Product for cosine similarity on normalized features
                # Higher scores = more similar (correct ranking)
                self.index = faiss.IndexFlatIP(self.features.shape[1])
            else:
                # Use L2 distance for Euclidean similarity
                # Lower scores = more similar
                self.index = faiss.IndexFlatL2(self.features.shape[1])

            self.index.add(self.features)
            print(f"Built {'cosine' if self.use_cosine else 'L2'} similarity index")

        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {e}")

    def query(self, img, k=5):
        """Enhanced query with proper similarity handling."""
        if k <= 0:
            raise ValueError("k must be positive")

        # Ensure k doesn't exceed database size
        k = min(k, len(self.fns))

        try:
            # Extract and process query feature
            q = self.feature_extractor.extract(img)
            if isinstance(q, torch.Tensor):
                q = q.detach().cpu().numpy()
            q = q.astype('float32').flatten()

            # Check for invalid query feature
            if np.any(np.isnan(q)) or np.any(np.isinf(q)) or q.size == 0:
                raise ValueError("Invalid query feature extracted")

            # Normalize if using cosine similarity
            if self.use_cosine:
                q_norm = np.linalg.norm(q)
                if q_norm == 0:
                    raise ValueError("Zero-norm query feature")
                q = q / q_norm

            # Search - note that for IndexFlatIP, higher scores are better
            scores, indices = self.index.search(q.reshape(1, -1), k)

            if self.use_cosine:
                # For cosine similarity, convert back to distances (lower = more similar)
                # Cosine distance = 1 - cosine_similarity
                distances = 1.0 - scores[0]
            else:
                # For L2, scores are already distances
                distances = scores[0]

            # Return results - filenames without extensions
            result_names = [self.fns[i] for i in indices[0]]
            return distances, result_names

        except Exception as e:
            print(f"Query failed: {e}")
            # Return failed query indication instead of fake results
            raise RuntimeError(f"Query processing failed: {e}")

    def get_database_info(self):
        """Get information about the loaded database."""
        return {
            'num_images': len(self.fns),
            'feature_dim': self.features.shape[1] if hasattr(self, 'features') else 0,
            'similarity_metric': 'cosine' if self.use_cosine else 'L2',
            'sample_filenames': self.fns[:10] if self.fns else []
        }


class QueryHandlerGPU(QueryHandler):
    """GPU version with proper resource management and similarity handling."""

    def __init__(self, feature_extractor, feature_dir, gpu_id=0, use_cosine=True):
        self.gpu_id = gpu_id
        self.res = None
        super().__init__(feature_extractor, feature_dir, use_cosine)

    def _build_index(self):
        """Build GPU index with proper similarity metric."""
        try:
            # Initialize GPU resources
            self.res = faiss.StandardGpuResources()

            # Create appropriate CPU index first
            if self.use_cosine:
                cpu_index = faiss.IndexFlatIP(self.features.shape[1])
            else:
                cpu_index = faiss.IndexFlatL2(self.features.shape[1])

            # Move to GPU
            self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index)
            self.index.add(self.features)

            print(f"Built GPU {'cosine' if self.use_cosine else 'L2'} similarity index on GPU {self.gpu_id}")

        except Exception as e:
            if self.res:
                self.res.noTempMemory()
                self.res = None
            raise RuntimeError(f"Failed to build GPU index: {e}")

    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'res') and self.res:
            try:
                self.res.noTempMemory()
            except:
                pass


# Factory function for easy instantiation
def create_query_handler(feature_extractor, feature_dir, use_gpu=False, gpu_id=0, use_cosine=True):
    """
    Factory function to create appropriate query handler.

    Args:
        feature_extractor: Feature extraction model
        feature_dir: Directory containing .pt feature files
        use_gpu: Whether to use GPU acceleration
        gpu_id: GPU device ID (if use_gpu=True)
        use_cosine: Whether to use cosine similarity (True) or L2 distance (False)

    Returns:
        QueryHandler or QueryHandlerGPU instance
    """
    if use_gpu:
        try:
            return QueryHandlerGPU(feature_extractor, feature_dir, gpu_id, use_cosine)
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU implementation")
            return QueryHandler(feature_extractor, feature_dir, use_cosine)
    else:
        return QueryHandler(feature_extractor, feature_dir, use_cosine)


# Example usage:
if __name__ == "__main__":
    # Test the query handler
    from preparation.feature_extractor import mmpretrain_resnet50_extractor

    # Initialize
    feature_extractor = mmpretrain_resnet50_extractor()
    query_handler = create_query_handler(
        feature_extractor,
        "/path/to/features",
        use_gpu=True,  # Try GPU first
        use_cosine=True  # Use cosine similarity
    )

    # Print database info
    info = query_handler.get_database_info()
    print("Database Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")