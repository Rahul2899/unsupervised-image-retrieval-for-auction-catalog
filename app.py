import gradio as gr
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from glob import glob
import time
import logging
import random

# Try to use FAISS if available, otherwise fall back to sklearn
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")

# Configuration paths - adjust these based on your setup
FEATURES_BASE = "<LOCAL_USER>/Downloads/APP/features_c4_c5_average_new"
METHOD_NAME = "C4_C5_average_gem_c1_pca1024_pytorch"
IMAGE_DIRS = {
    "660K": "<LOCAL_USER>/Downloads/APP/total_new_crops"
}
QUERY_DIR = "<LOCAL_USER>/Downloads/APP/query_images"

# Model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
TOP_K_OPTIONS = [10, 20, 50, 100]
LAYERS = ["C4", "C5"]
FUSION_METHOD = "average"
POOLING = "gem"
PCA_DIM = 1024
CACHE_SIZE_LIMIT = 3

# Simple cache to avoid reloading features repeatedly
class MemoryEfficientCache:
    def __init__(self, max_size=CACHE_SIZE_LIMIT):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.info(f"Evicted {lru_key} from cache")
        
        self.cache[key] = value
        self.access_order.append(key)
        logger.info(f"Cached {key}")
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()

feature_cache = MemoryEfficientCache()

# GeM pooling layer
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = eps
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1./self.p)

# DINO model wrapper with feature hooks
class HookBasedDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.encoder.fc = nn.Identity()
        self.features = {}
        
        def hook(name):
            def fn(module, input, output):
                self.features[name] = output.detach().clone()
            return fn
        
        # Grab features from layer3 (C4) and layer4 (C5)
        self.encoder.layer3.register_forward_hook(hook('C4'))
        self.encoder.layer4.register_forward_hook(hook('C5'))

    def forward(self, x):
        self.features = {}
        _ = self.encoder(x)
        return self.features

def pool_features(feat_map, pooling='gem'):
    """Apply pooling to feature maps"""
    if pooling == 'gem':
        pooled = GeM(p=3)(feat_map).flatten(1)
    else:
        raise ValueError(f"Only GeM pooling supported, got: {pooling}")
    
    pooled = pooled + 1e-8
    return pooled

def apply_consistent_average_fusion_for_queries(c4_feat, c5_feat, device='cpu'):
    """Fuse C4 and C5 features using average method - must match training"""
    
    # Normalize each layer
    c4_norm = F.normalize(c4_feat, p=2, dim=1, eps=1e-8)
    c5_norm = F.normalize(c5_feat, p=2, dim=1, eps=1e-8)
    
    # Handle dimension mismatch (C4 is 1024, C5 is 2048)
    c4_dim = c4_norm.shape[1]
    c5_dim = c5_norm.shape[1]
    
    if c4_dim != c5_dim:
        if c5_dim == 2 * c4_dim:
            # Just repeat C4 features to match
            c4_expanded = torch.cat([c4_norm, c4_norm], dim=1)
        else:
            # Pad with zeros
            padding_size = c5_dim - c4_dim
            c4_expanded = F.pad(c4_norm, (0, padding_size), 'constant', 0)
    else:
        c4_expanded = c4_norm
    
    # Simple average
    fused = (c4_expanded + c5_norm) / 2.0
    fused = F.normalize(fused, p=2, dim=1, eps=1e-8)
    
    return fused

def apply_pca_from_dict(pca_data, X):
    """Apply PCA using saved model"""
    if pca_data is None:
        logger.warning("No PCA model provided, returning original features")
        return X
    
    try:
        X = X.astype(np.float32)
        
        # Handle both dict and sklearn formats
        if isinstance(pca_data, dict):
            mean = pca_data.get('mean_', np.zeros(X.shape[1]))
            components = pca_data.get('components_', np.eye(X.shape[1]))
            
            X_centered = X - mean.reshape(1, -1)
            
            # Check dimensions
            if components.shape[1] != X_centered.shape[1]:
                logger.warning(f"Dimension mismatch: components {components.shape} vs features {X_centered.shape}")
                min_dim = min(components.shape[1], X_centered.shape[1])
                components = components[:, :min_dim]
                X_centered = X_centered[:, :min_dim]
            
            X_transformed = X_centered @ components.T
            
            # Whitening if specified
            if pca_data.get('whiten', False) and 'explained_variance_' in pca_data:
                explained_variance = pca_data['explained_variance_']
                X_transformed = X_transformed / (np.sqrt(explained_variance) + 1e-10)
        else:
            X_transformed = pca_data.transform(X)
        
        logger.info(f"PCA applied: {X.shape} -> {X_transformed.shape}")
        return X_transformed
        
    except Exception as e:
        logger.error(f"PCA application failed: {e}")
        logger.warning("Returning original features")
        return X

# C1 preprocessing - this is a specific pipeline from the paper
def preprocess_c1_with_steps(img_input):
    """Apply C1 preprocessing with step tracking"""
    try:
        # Handle different input types
        if isinstance(img_input, np.ndarray):
            original_img = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
        elif isinstance(img_input, Image.Image):
            original_img = img_input.convert("RGB")
        else:
            logger.warning(f"Unexpected input type: {type(img_input)}")
            return None, []
        
        steps = []
        
        steps.append({
            'name': 'Step 0: Original Input',
            'image': original_img.copy(),
            'description': 'Original image as provided by user',
            'technical': f'Size: {original_img.size}, Mode: {original_img.mode}'
        })
        
        img = np.array(original_img)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        steps.append({
            'name': 'Step 1: RGB to BGR',
            'image': Image.fromarray(img_bgr_display),
            'description': 'Convert to BGR format for OpenCV processing',
            'technical': f'OpenCV requires BGR color order. Shape: {img_bgr.shape}'
        })
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray_rgb_display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        steps.append({
            'name': 'Step 2: Convert to Grayscale',
            'image': Image.fromarray(img_gray_rgb_display),
            'description': 'Remove color information to focus on structure',
            'technical': f'Single channel grayscale. Shape: {img_gray.shape}. Removes color bias.'
        })
        
        # Back to RGB (3 channels)
        img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        steps.append({
            'name': 'Step 3: Grayscale to RGB',
            'image': Image.fromarray(img_gray_rgb),
            'description': 'Convert back to 3-channel RGB format',
            'technical': f'Shape: {img_gray_rgb.shape}. Creates RGB with identical R=G=B values.'
        })
        
        # Rotate if portrait
        h, w = img_gray_rgb.shape[:2]
        rotation_applied = False
        if h > w:
            img_rotated = cv2.rotate(img_gray_rgb, cv2.ROTATE_90_CLOCKWISE)
            rotation_applied = True
            rotation_desc = f'Applied 90° clockwise rotation'
        else:
            img_rotated = img_gray_rgb.copy()
            rotation_desc = f'No rotation needed (already landscape)'
        
        steps.append({
            'name': 'Step 4: Rotation Correction',
            'image': Image.fromarray(img_rotated),
            'description': f'Ensure landscape orientation. {rotation_desc}',
            'technical': f'Original: {h}x{w} → Final: {img_rotated.shape[0]}x{img_rotated.shape[1]}. Rotated: {rotation_applied}'
        })
        
        # Apply denoising
        try:
            img_denoised = cv2.fastNlMeansDenoisingColored(img_rotated, None, 3, 3, 7, 15)
            denoising_success = True
            denoising_desc = 'Applied non-local means denoising'
        except Exception as e:
            img_denoised = img_rotated.copy()
            denoising_success = False
            denoising_desc = f'Denoising failed: {str(e)}'
        
        steps.append({
            'name': 'Step 5: Noise Reduction',
            'image': Image.fromarray(img_denoised),
            'description': f'{denoising_desc}',
            'technical': f'Non-local means denoising. Success: {denoising_success}. Parameters: h=3, hColor=3, templateWindowSize=7, searchWindowSize=15'
        })
        
        final_img = Image.fromarray(img_denoised)
        steps.append({
            'name': 'Final Result',
            'image': final_img,
            'description': 'Preprocessed image ready for DINO feature extraction',
            'technical': f'Complete C1 pipeline applied. Final size: {final_img.size}. Ready for ResNet-50 processing.'
        })
        
        return final_img, steps
        
    except Exception as e:
        logger.error(f"Step-by-step preprocessing failed: {e}")
        try:
            if isinstance(img_input, Image.Image):
                error_img = img_input.convert("RGB")
            elif isinstance(img_input, np.ndarray):
                error_img = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
            else:
                error_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
            
            error_steps = [{
                'name': 'Error in Processing', 
                'image': error_img, 
                'description': f'Processing failed: {str(e)}', 
                'technical': 'Using original image without preprocessing'
            }]
            return error_img, error_steps
        except:
            return None, []

def preprocess_c1_pil(img_input):
    """Quick C1 preprocessing without step tracking"""
    try:
        final_img, _ = preprocess_c1_with_steps(img_input)
        return final_img if final_img else img_input
    except Exception as e:
        logger.warning(f"C1 preprocess failed, using original image: {e}")
        if isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        else:
            return Image.new("RGB", (224, 224), color="white")

# Standard image transformations
C1_TRANSFORM = transforms.Compose([
    transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model singleton pattern
_cached_model = None

def get_model():
    """Get or create model instance"""
    global _cached_model
    if _cached_model is None:
        logger.info("Loading DINO model...")
        _cached_model = HookBasedDINO().to(DEVICE).eval()
        logger.info(f"Model loaded on {DEVICE}")
    return _cached_model

def get_query_images():
    """Find all query images in the query directory"""
    if not os.path.exists(QUERY_DIR):
        logger.warning(f"Query directory not found: {QUERY_DIR}")
        return []
    
    query_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
        query_images.extend(glob(os.path.join(QUERY_DIR, ext)))
        query_images.extend(glob(os.path.join(QUERY_DIR, ext.upper())))
    
    query_images.sort()
    logger.info(f"Found {len(query_images)} query images in {QUERY_DIR}")
    return query_images

def load_query_image(image_path):
    """Load a single query image"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load query image {image_path}: {e}")
        return None

def get_available_datasets():
    """Check what datasets we have features for"""
    if not os.path.exists(FEATURES_BASE):
        logger.error(f"Features base directory not found: {FEATURES_BASE}")
        return []
    
    method_dir = os.path.join(FEATURES_BASE, METHOD_NAME)
    if not os.path.exists(method_dir):
        logger.error(f"Method directory not found: {method_dir}")
        return []
    
    available = []
    for item in os.listdir(method_dir):
        item_path = os.path.join(method_dir, item)
        if os.path.isdir(item_path):
            features_file = os.path.join(item_path, "features.pt")
            index_file = os.path.join(item_path, "index.json")
            if os.path.exists(features_file) and os.path.exists(index_file):
                try:
                    _ = torch.load(features_file, map_location="cpu", weights_only=True)
                    with open(index_file, 'r') as f:
                        _ = json.load(f)
                    available.append(item)
                    logger.info(f"Found valid dataset: {item}")
                except Exception as e:
                    logger.warning(f"Invalid dataset {item}: {e}")
    
    return sorted(available)

def load_consolidated_db_features(dataset_name):
    """Load precomputed features for a dataset"""
    cache_key = f"{dataset_name}_{METHOD_NAME}"
    
    # Check if we already have it in memory
    cached_data = feature_cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached data for {dataset_name}")
        return cached_data
    
    feat_dir = os.path.join(FEATURES_BASE, METHOD_NAME, dataset_name)
    features_file = os.path.join(feat_dir, "features.pt")
    index_file = os.path.join(feat_dir, "index.json")
    pca_file = os.path.join(feat_dir, "pca_model.pkl")

    if not (os.path.exists(features_file) and os.path.exists(index_file)):
        logger.error(f"Missing features in {feat_dir}")
        return np.empty((0, PCA_DIM), dtype="float32"), [], None

    try:
        feats_t = torch.load(features_file, map_location="cpu", weights_only=True)
        feats = feats_t.detach().cpu().numpy().astype("float32")

        with open(index_file, "r") as f:
            idx_map = json.load(f)["index_mapping"]

        # Build ordered list of filenames
        names = [""] * len(idx_map)
        for stem, i in idx_map.items():
            if i < len(names):
                names[i] = stem

        # Load PCA if it exists
        pca_model = None
        if os.path.exists(pca_file):
            with open(pca_file, 'rb') as f:
                pca_model = pickle.load(f)

        data = (feats, names, pca_model)
        feature_cache.put(cache_key, data)
        
        logger.info(f"Loaded DB features: {feats.shape} from {feat_dir}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return np.empty((0, PCA_DIM), dtype="float32"), [], None

def find_image_by_stem(root: str, stem: str) -> str:
    """Try to find the actual image file given a stem name"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(root, stem + ext)
        if os.path.exists(p):
            return p
    
    # Last resort
    import glob
    matches = glob.glob(os.path.join(root, stem) + ".*")
    return matches[0] if matches else ""

def normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize vectors"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def search_topk_cosine_only(q: np.ndarray, db: np.ndarray, k: int):
    """Find top k most similar vectors using cosine similarity"""
    if db.size == 0:
        return None, None
    k = min(int(k), db.shape[0])
    qn, dn = normalize(q.astype("float32")), normalize(db.astype("float32"))
    
    if FAISS_OK:
        # Use FAISS for fast inner product search
        index = faiss.IndexFlatIP(dn.shape[1])
        index.add(dn)
        D, I = index.search(qn, k)
        return D[0], I[0]
    else:
        # Fallback to sklearn
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(qn, dn)[0]
        idx = np.argsort(sims)[::-1][:k]
        return sims[idx], idx

@torch.no_grad()
def encode_query(image_input, model: nn.Module, pca_model=None) -> np.ndarray:
    """Extract features from query image"""
    try:
        # Make sure we have a PIL Image
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            logger.error(f"Unsupported image type: {type(image_input)}")
            return None
        
        # Preprocess
        x_img = preprocess_c1_pil(image)
        x = C1_TRANSFORM(x_img).unsqueeze(0).to(DEVICE)
        
        # Extract features
        features_dict = model(x)
        
        # Pool features
        c4_feat = pool_features(features_dict['C4'].cpu(), pooling=POOLING)
        c5_feat = pool_features(features_dict['C5'].cpu(), pooling=POOLING)
        
        # Fuse layers
        fused_features = apply_consistent_average_fusion_for_queries(c4_feat, c5_feat, device='cpu')
        
        # Apply PCA if we have it
        if pca_model is not None:
            fused_np = fused_features.numpy()
            pca_features = apply_pca_from_dict(pca_model, fused_np)
            final_features = torch.from_numpy(pca_features).float()
        else:
            final_features = fused_features
        
        # Final normalization
        final_features = F.normalize(final_features, p=2, dim=1, eps=1e-8)
        
        return final_features.cpu().numpy().astype("float32")
    
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        return None

def rotate_image(image, angle):
    """Simple image rotation"""
    if image is None:
        return None
    return image.rotate(angle, expand=True)

def search_with_scores(image_input, dataset_name: str, topk: int):
    """Main search function"""
    if image_input is None:
        return [], "Please upload an image or select a preloaded query first."
    
    try:
        model = get_model()
        db_feats, db_stems, pca_model = load_consolidated_db_features(dataset_name)
        
        if db_feats.size == 0:
            return [], f"❌ Failed to load database features for {dataset_name}"
        
        if dataset_name not in IMAGE_DIRS:
            return [], f"❌ Dataset {dataset_name} not configured in IMAGE_DIRS"
        
        db_root = IMAGE_DIRS[dataset_name]
        
        start_time = time.time()
        q = encode_query(image_input, model, pca_model)
        if q is None:
            return [], "❌ Failed to encode query image"
        
        sims, idxs = search_topk_cosine_only(q, db_feats, topk)
        if sims is None:
            return [], "❌ Search failed"
        
        # Build results with captions
        results = []
        for rank, (i, s) in enumerate(zip(idxs, sims), 1):
            stem = db_stems[i] if i < len(db_stems) else f"unknown_{i}"
            p = find_image_by_stem(db_root, stem)
            
            score_color = "🟢" if s > 0.8 else "🟡" if s > 0.6 else "🟠" if s > 0.4 else "🔴"
            caption = f"#{rank:02d} • {score_color} Score: {float(s):.4f}\n📁 {stem}"
            
            if p:
                results.append((p, caption))
            else:
                logger.warning(f"Image not found: {stem}")
        
        search_time = time.time() - start_time
        avg_score = np.mean(sims) if len(sims) > 0 else 0
        status = f"""✅ **Search Complete!**
📊 Found {len(results)} results from **{dataset_name}** dataset
⏱️ Search time: {search_time:.2f}s | 📈 Avg similarity: {avg_score:.4f}
🗃️ Database size: {len(db_stems):,} images | 🎯 Similarity: Cosine only"""
        
        return results, status
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return [], f"❌ Search error: {str(e)}"

def process_and_show_query_detailed(image_input):
    """Show detailed preprocessing steps"""
    if image_input is None:
        return None, None, "", [], ""
    
    try:
        if isinstance(image_input, np.ndarray):
            original_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            original_img = image_input
        else:
            logger.error(f"Unsupported image type: {type(image_input)}")
            return image_input, image_input, f"❌ **Unsupported image type**: {type(image_input)}", [], ""
        
        processed_img, steps = preprocess_c1_with_steps(original_img)
        
        step_gallery = []
        step_descriptions = []
        
        for i, step in enumerate(steps):
            step_gallery.append((step['image'], f"{step['name']}\n{step['description']}"))
            step_descriptions.append(f"**{step['name']}**\n- {step['description']}\n- {step['technical']}\n")
        
        status = f"""✅ **C1 Preprocessing Pipeline Complete**
📊 **Total Steps**: {len(steps)}
🔄 **Pipeline**: Original → BGR → Grayscale → RGB → Rotation → Denoising → Final

**Purpose**: Normalize images for consistent DINO ResNet-50 feature extraction"""
        
        technical_summary = "\n".join(step_descriptions)
        
        return original_img, processed_img, status, step_gallery, technical_summary
        
    except Exception as e:
        logger.error(f"Detailed processing failed: {e}")
        if isinstance(image_input, Image.Image):
            return image_input, image_input, f"❌ **Processing failed**: {str(e)}", [], ""
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
            return img, img, f"❌ **Processing failed**: {str(e)}", [], ""
        else:
            return None, None, f"❌ **Processing failed**: {str(e)}", [], ""

def process_and_show_query(image_input):
    """Simplified processing preview"""
    if image_input is None:
        return None, None, "⚠️ Please upload an image or select a preloaded query first."
    
    original, processed, status, _, _ = process_and_show_query_detailed(image_input)
    return original, processed, status

def get_random_query():
    """Pick a random query image"""
    query_images = get_query_images()
    if not query_images:
        return None
    
    random_path = random.choice(query_images)
    return load_query_image(random_path)

def create_enhanced_interface():
    """Build the Gradio UI"""
    available_datasets = get_available_datasets()
    query_images = get_query_images()
    
    if not available_datasets:
        logger.error("No datasets available!")
        def error_fn(*args):
            return [], "❌ No datasets found! Please check feature extraction setup."
        
        with gr.Blocks(title="❌ C4+C5 Retrieval - Setup Error") as demo:
            gr.Markdown("# ❌ C4+C5 Average Fusion Retrieval - Setup Error")
            gr.Markdown("No feature databases found. Please run the feature extraction script first.")
            
        return demo
    
    with gr.Blocks(
        title="🔍 Advanced Image Retrieval System", 
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate"
        ),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .feature-card {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .status-success {
            border-left: 4px solid #10b981;
            background-color: #f0fdf4;
            color: #166534;
            padding: 12px;
            border-radius: 6px;
        }
        .status-info {
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            color: #1e40af;
            padding: 12px;
            border-radius: 6px;
        }
        .technical-details {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 12px;
            color: #334155;
        }
        """
    ) as demo:
        
        gr.Markdown("# 🔍 Advanced C4+C5 Fusion Image Retrieval", elem_classes=["main-header"])
        gr.Markdown("""
        <div class="feature-card">
        <b>🚀 Enhanced Visual Similarity Search</b> powered by DINO ResNet-50 with C4+C5 layer fusion.<br/>
        Upload your own image or select from preloaded queries to find visually similar images.
        </div>
        """)
        
        with gr.Row():
            # Left side - query controls
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Query Selection")
                
                with gr.Tabs():
                    with gr.Tab("📤 Upload Image"):
                        uploaded_image = gr.Image(
                            type="pil", 
                            label="Upload Your Query Image", 
                            height=300,
                            show_label=True
                        )
                    
                    with gr.Tab("📚 Preloaded Queries"):
                        if query_images:
                            query_names = [os.path.basename(img) for img in query_images]
                            query_dropdown = gr.Dropdown(
                                choices=query_names,
                                label=f"Select Query ({len(query_images)} available)",
                                info="Choose from preloaded query images"
                            )
                            
                            with gr.Row():
                                load_query_btn = gr.Button("📋 Load Selected", variant="secondary")
                                random_query_btn = gr.Button("🎲 Random Query", variant="secondary")
                        else:
                            gr.Markdown("⚠️ No query images found in the specified directory.")
                    
                    with gr.Tab("👁️ Processing Preview"):
                        gr.Markdown("**Preview how your query image will be processed:**")
                        with gr.Row():
                            original_preview = gr.Image(label="🔸 Original", height=200, interactive=False)
                            processed_preview = gr.Image(label="🔹 C1 Processed", height=200, interactive=False)
                        
                        process_status = gr.Markdown("", elem_classes=["status-info"])
                        preview_btn = gr.Button("👁️ Preview Processing", variant="secondary")
                    
                    with gr.Tab("🔬 Step-by-Step Analysis"):
                        gr.Markdown("**Detailed preprocessing pipeline visualization:**")
                        detailed_preview_btn = gr.Button("🔍 Analyze Processing Steps", variant="primary")
                        
                        steps_gallery = gr.Gallery(
                            label="Preprocessing Steps", 
                            columns=3, 
                            rows=2,
                            height=400,
                            preview=True,
                            object_fit="contain",
                            show_share_button=True
                        )
                        
                        technical_details = gr.Markdown("", elem_classes=["technical-details"])
                
                gr.Markdown("### 🖼️ Current Query")
                current_query = gr.Image(label="Active Query Image", height=250, interactive=False)
                
                gr.Markdown("### ⚙️ Search Configuration")
                with gr.Row():
                    dataset_choice = gr.Dropdown(
                        choices=available_datasets,
                        value=available_datasets[0],
                        label="📂 Target Dataset",
                        info=f"{len(available_datasets)} datasets available"
                    )
                    
                    topk_slider = gr.Slider(
                        minimum=5, 
                        maximum=100, 
                        value=20, 
                        step=5, 
                        label="🔢 Results Count",
                        info="Number of similar images to retrieve"
                    )
                
                search_btn = gr.Button(
                    "🔍 Search Similar Images", 
                    variant="primary", 
                    size="lg",
                    elem_id="search-button"
                )
                
                search_status = gr.Markdown("", elem_classes=["status-info"])
            
            # Right side - results
            with gr.Column(scale=3):
                gr.Markdown("### 🎯 Search Results")
                
                with gr.Tabs():
                    with gr.Tab("🖼️ Results Gallery"):
                        results_gallery = gr.Gallery(
                            label="Similar Images with Scores", 
                            columns=4, 
                            rows=4,
                            height=700,
                            preview=True,
                            object_fit="contain",
                            show_share_button=False
                        )
                    
                    with gr.Tab("🔄 Rotation Tools"):
                        gr.Markdown("**Rotate result images for better viewing:**")
                        
                        with gr.Row():
                            with gr.Column():
                                selected_idx = gr.Number(
                                    value=0, 
                                    label="Image Index",
                                    info="0-based index of result image",
                                    precision=0
                                )
                                rotate_angle = gr.Slider(
                                    -180, 180, 
                                    value=0, 
                                    step=90, 
                                    label="Rotation Angle (degrees)"
                                )
                            
                            with gr.Column():
                                rotate_btn = gr.Button("🔄 Rotate Selected", variant="secondary")
                                reset_rotation_btn = gr.Button("↩️ Reset", variant="outline")
                        
                        rotated_image = gr.Image(
                            label="Rotated Result Image", 
                            height=400,
                            show_label=True
                        )
                    
                    with gr.Tab("📊 Analysis"):
                        gr.Markdown("**Search Performance Metrics:**")
                        analysis_output = gr.Markdown("")

        with gr.Accordion("🔬 Technical Information", open=False):
            gr.Markdown(f"""
            ### 🏗️ **Architecture Details**
            - **🤖 Model**: DINO ResNet-50 (Self-supervised Learning)
            - **🧠 Layers**: C4 (1024D) + C5 (2048D) with deterministic expansion
            - **🔗 Fusion**: L2-normalized average fusion
            - **🏊 Pooling**: GeM (Generalized Mean Pooling, p=3)
            - **📉 Reduction**: PCA to {PCA_DIM}D with whitening
            - **🔄 Preprocessing**: C1 pipeline (grayscale → rotation → denoising)
            - **📏 Similarity**: Cosine similarity only (no Euclidean)
            - **⚡ Acceleration**: FAISS indexing when available
            
            ### 📁 **Dataset Configuration**
            - **Available Datasets**: {', '.join(available_datasets)}
            - **Query Directory**: `{QUERY_DIR}`
            - **Features Method**: `{METHOD_NAME}`
            - **Device**: {DEVICE} {'🚀' if DEVICE == 'cuda' else '💻'}
            - **FAISS Status**: {'✅ Available' if FAISS_OK else '❌ Using sklearn fallback'}
            
            ### 🔍 **Search Process**
            1. **Image Upload/Selection** → C1 preprocessing pipeline
            2. **Feature Extraction** → DINO C4+C5 layers with GeM pooling  
            3. **Fusion** → Average fusion with dimension alignment
            4. **Dimensionality Reduction** → PCA transformation
            5. **Similarity Search** → Cosine similarity ranking
            6. **Results** → Top-K retrieval with confidence scores
            """)
        
        # Wire up all the event handlers
        def update_current_query_from_upload(image):
            if image is not None:
                original, processed, status = process_and_show_query(image)
                return image, original, processed, status
            return None, None, None, ""
        
        def load_selected_query(query_name):
            if not query_name or not query_images:
                return None, None, None, ""
            
            query_path = None
            for img_path in query_images:
                if os.path.basename(img_path) == query_name:
                    query_path = img_path
                    break
            
            if query_path:
                image = load_query_image(query_path)
                if image:
                    original, processed, status = process_and_show_query(image)
                    return image, original, processed, status
            
            return None, None, None, f"❌ Failed to load {query_name}"
        
        def load_random_query():
            image = get_random_query()
            if image:
                original, processed, status = process_and_show_query(image)
                return image, original, processed, status
            return None, None, None, "❌ No query images available"
        
        uploaded_image.change(
            fn=update_current_query_from_upload,
            inputs=[uploaded_image],
            outputs=[current_query, original_preview, processed_preview, process_status]
        )
        
        if query_images:
            load_query_btn.click(
                fn=load_selected_query,
                inputs=[query_dropdown],
                outputs=[current_query, original_preview, processed_preview, process_status]
            )
            
            random_query_btn.click(
                fn=load_random_query,
                inputs=[],
                outputs=[current_query, original_preview, processed_preview, process_status]
            )
        
        preview_btn.click(
            fn=process_and_show_query,
            inputs=[current_query],
            outputs=[original_preview, processed_preview, process_status]
        )
        
        detailed_preview_btn.click(
            fn=process_and_show_query_detailed,
            inputs=[current_query],
            outputs=[original_preview, processed_preview, process_status, steps_gallery, technical_details]
        )
        
        search_btn.click(
            fn=search_with_scores,
            inputs=[current_query, dataset_choice, topk_slider],
            outputs=[results_gallery, search_status]
        )
        
        def rotate_result_image(gallery_data, idx, angle):
            try:
                if not gallery_data or idx >= len(gallery_data) or idx < 0:
                    return None, "❌ Invalid image index"
                
                image_path = gallery_data[int(idx)][0]
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    rotated = img.rotate(angle, expand=True)
                    return rotated, f"✅ Rotated image #{int(idx)} by {angle}°"
                return None, f"❌ Image file not found: {image_path}"
            except Exception as e:
                logger.error(f"Rotation failed: {e}")
                return None, f"❌ Rotation failed: {str(e)}"
        
        def reset_rotation():
            return 0
        
        rotate_btn.click(
            fn=rotate_result_image,
            inputs=[results_gallery, selected_idx, rotate_angle],
            outputs=[rotated_image, analysis_output]
        )
        
        reset_rotation_btn.click(
            fn=reset_rotation,
            outputs=[rotate_angle]
        )
        
        def update_analysis(gallery_data, status_text):
            if not gallery_data:
                return "📊 **No search results to analyze**"
            
            num_results = len(gallery_data)
            analysis = f"""
📊 **Search Analysis:**
- **Results Found**: {num_results} images
- **Search Method**: Cosine similarity (C4+C5 fusion)
- **Feature Dimension**: {PCA_DIM}D (after PCA)
- **Processing Pipeline**: C1 → DINO → GeM → Fusion → PCA → Search

🎯 **Quality Indicators**:
- Green 🟢: High similarity (>0.8)
- Yellow 🟡: Good similarity (>0.6)  
- Orange 🟠: Moderate similarity (>0.4)
- Red 🔴: Low similarity (<0.4)

💡 **Tips**: 
- Try rotating images in the rotation tab for better viewing
- Higher scores indicate better matches
- Results are ranked by similarity score
"""
            return analysis
        
        results_gallery.change(
            fn=update_analysis,
            inputs=[results_gallery, search_status],
            outputs=[analysis_output]
        )
        
        gr.Markdown("### 🚀 Quick Start")
        gr.Markdown("""
        1. **📤 Upload an image** or **📚 select a preloaded query**
        2. **👁️ Preview processing** to see how your image will be transformed
        3. **🔍 Configure search** settings (dataset and result count)
        4. **🎯 Search** for similar images
        5. **🔄 Rotate results** if needed for better viewing
        """)
        
        if query_images:
            demo.load(
                fn=load_random_query,
                outputs=[current_query, original_preview, processed_preview, process_status]
            )
    
    return demo

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔍 ADVANCED C4+C5 FUSION IMAGE RETRIEVAL SYSTEM")
    logger.info("=" * 60)
    logger.info(f"🖥️  Device: {DEVICE}")
    logger.info(f"⚡ FAISS available: {FAISS_OK}")
    logger.info(f"📁 Features base: {FEATURES_BASE}")
    logger.info(f"🔬 Method: {METHOD_NAME}")
    logger.info(f"🎯 Query directory: {QUERY_DIR}")
    
    for name, path in IMAGE_DIRS.items():
        status = "✅" if os.path.exists(path) else "❌"
        logger.info(f"🗃️  Image dir {name}: {status} {path}")
    
    query_status = "✅" if os.path.exists(QUERY_DIR) else "❌"
    query_count = len(get_query_images()) if query_status == "✅" else 0
    logger.info(f"🔍 Query dir: {query_status} {QUERY_DIR} ({query_count} images)")
    
    available_datasets = get_available_datasets()
    logger.info(f"📊 Available datasets: {len(available_datasets)} - {available_datasets}")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting enhanced Gradio interface...")
    
    demo = create_enhanced_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=list(IMAGE_DIRS.values()) + [FEATURES_BASE, QUERY_DIR]
    )