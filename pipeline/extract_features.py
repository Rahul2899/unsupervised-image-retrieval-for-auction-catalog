#!/usr/bin/env python3
"""
Feature Extraction — DINO ResNet50 - 660K Dataset - PCA-512 with Whitening ONLY
MEMORY OPTIMIZED VERSION - Fixes GPU OOM errors
Run: python extractfull.py [--overwrite]

Configurations:
1. C4_C5_max_mac_c1_pca512_whiten (C4+C5 fusion, MAC pooling, PCA-512, whitening)
2. C5_mac_c1_pca512_whiten (C5 only, MAC pooling, PCA-512, whitening)

Key optimizations:
- Features stored on CPU immediately during extraction
- Chunked concatenation for large datasets
- PCA processing on CPU to avoid GPU OOM errors
"""

import os, json, logging, cv2, pickle, sys, gc
from glob import glob
from tqdm import tqdm
import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from sklearn.decomposition import IncrementalPCA

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DINO_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "crops_yolo_v2"))
FEATURES_BASE = os.environ.get("DINO_FEATURES_BASE", os.path.join(PROJECT_ROOT, "data", "features_yolo_v2"))
BATCH_SIZE = int(os.environ.get("DINO_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("DINO_NUM_WORKERS", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.tif","*.webp")

os.makedirs(FEATURES_BASE, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dino_660k_pca512")

# ============================================================================
# CONFIGURATIONS - ONLY PCA-512 WITH WHITENING
# ============================================================================
CONFIGS = {
    "C4_C5_max_mac_c1_pca512_whiten": {
        'layers': ['C4', 'C5'],
        'fusion': 'max',
        'pooling': 'mac',
        'pca_dim': 512,
        'pca_whiten': True
    },
    "C5_mac_c1_pca512_whiten": {
        'layers': ['C5'],
        'fusion': 'none',
        'pooling': 'mac',
        'pca_dim': 512,
        'pca_whiten': True
    }
}

# ============================================================================
# GPU HELPERS
# ============================================================================
def log_gpu_memory():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated()/1024**3
        r = torch.cuda.memory_reserved()/1024**3
        logger.info(f"🔧 GPU Memory - Allocated: {a:.2f}GB, Reserved: {r:.2f}GB")

def log_system_memory():
    """Log system RAM usage"""
    mem = psutil.virtual_memory()
    logger.info(f"💾 RAM: {mem.used/1024**3:.2f}GB / {mem.total/1024**3:.2f}GB "
                f"({mem.percent:.1f}% used, {mem.available/1024**3:.2f}GB available)")

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================================
# GeM POOLING
# ============================================================================
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1,1)).pow(1./self.p)

# ============================================================================
# DINO BACKBONE
# ============================================================================
class HookBasedDINO(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        logger.info("Loading DINO ResNet-50 from torch.hub...")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        self.backbone = model
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone = self.backbone.to(device)

        self.features = {}
        def hook(name):
            def fn(m, i, o):
                self.features[name] = o.detach().clone()
            return fn

        self.backbone.layer3.register_forward_hook(hook('C4'))
        self.backbone.layer4.register_forward_hook(hook('C5'))

        logger.info("✅ DINO backbone ready (C4/C5 hooks enabled)")

    def forward(self, x):
        self.features = {}
        _ = self.backbone(x.to(self.device))
        return self.features

# ============================================================================
# POOLING
# ============================================================================
def pool_features(feat_map, pooling='mac', device='cuda'):
    """Apply pooling to feature maps"""
    if pooling == 'gem':
        pooled = GeM(p=3).to(device)(feat_map).flatten(1)
    elif pooling == 'mac':
        pooled = F.adaptive_max_pool2d(feat_map, (1,1)).flatten(1)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return pooled + 1e-8

# ============================================================================
# NORMALIZATION
# ============================================================================
def l2_normalize(x):
    """L2 normalization"""
    return F.normalize(x, p=2, dim=1, eps=1e-8)

# ============================================================================
# VALIDATION - MEMORY OPTIMIZED
# ============================================================================
def validate_features(t, tag=""):
    """
    Validate feature tensor for NaN, Inf, and zero vectors
    OPTIMIZED: Works on CPU to save GPU memory
    """
    # Move to CPU for validation to avoid GPU OOM
    if t.is_cuda:
        t_check = t.cpu()
    else:
        t_check = t

    if torch.isnan(t_check).any():
        logger.error(f"❌ NaN detected in {tag}")
        return False
    if torch.isinf(t_check).any():
        logger.error(f"❌ Inf detected in {tag}")
        return False

    zero_count = (t_check.norm(dim=1) < 1e-6).sum().item()
    if zero_count > 0:
        logger.warning(f"⚠️ {zero_count} near-zero vectors in {tag}")

    std_val = t_check.std().item()
    mean_norm = t_check.norm(dim=1).mean().item()

    if std_val < 1e-4:
        logger.warning(f"⚠️ Very low variance ({std_val:.6f}) in {tag}")

    logger.info(f"✅ {tag}: shape={tuple(t.shape)}, std={std_val:.4f}, mean_norm={mean_norm:.4f}")
    return True

# ============================================================================
# PCA WITH WHITENING - 512D ONLY - MEMORY OPTIMIZED
# ============================================================================
def apply_pca_512_whiten(features_tensor, device='cuda', layer_info=""):
    """
    Apply PCA-512 with whitening - optimized for 660K dataset
    MEMORY OPTIMIZED: Processes on CPU to avoid GPU OOM
    """
    pca_dim = 512
    logger.info(f"🔄 PCA {layer_info}: {tuple(features_tensor.shape)} -> {pca_dim} (whiten=True)")
    log_gpu_memory()

    # CRITICAL: Move to CPU immediately for validation and processing
    logger.info("📤 Moving features to CPU for PCA processing...")
    features_cpu = features_tensor.cpu()
    del features_tensor
    clear_gpu_cache()

    if not validate_features(features_cpu, f"pre-PCA {layer_info}"):
        logger.warning("⚠️ Validation failed, continuing anyway...")

    # Convert to numpy (already on CPU)
    logger.info("Converting to numpy...")
    X = features_cpu.numpy().astype(np.float32)
    del features_cpu

    n, d = X.shape
    actual = min(pca_dim, min(n, d) - 1)

    if actual != pca_dim:
        logger.warning(f"⚠️ Reduce PCA dim {pca_dim}→{actual} due to data constraints")

    # Use IncrementalPCA for 660K
    logger.info(f"Using IncrementalPCA for 660K dataset (n={n:,}, d={d})")

    # Adaptive batch size based on dataset size
    if n > 500000:
        bs = 2048
    elif n > 100000:
        bs = 1024
    else:
        bs = 512

    logger.info(f"IncrementalPCA batch size: {bs}")

    ipca = IncrementalPCA(n_components=actual, whiten=True, batch_size=bs)

    # Fit
    logger.info("Fitting PCA...")
    for i in tqdm(range(0, n, bs), desc=f"PCA fit {layer_info}"):
        ipca.partial_fit(X[i:i+bs])

    # Transform
    logger.info("Transforming features...")
    parts = []
    for i in tqdm(range(0, n, bs), desc=f"PCA transform {layer_info}"):
        parts.append(ipca.transform(X[i:i+bs]))

    Xp = np.vstack(parts)
    del parts, X  # Free memory

    logger.info(f"📊 Total explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

    # Convert back to tensor and move to GPU
    logger.info(f"📥 Moving PCA results back to {device}...")
    Xt = torch.from_numpy(Xp.astype(np.float32)).to(device)
    del Xp  # Free numpy array

    validate_features(Xt, f"post-PCA {layer_info}")
    log_gpu_memory()

    return Xt, ipca

# ============================================================================
# FUSION - MEMORY OPTIMIZED (FULLY CPU-BASED)
# ============================================================================
def apply_fusion(all_layer_features, config, device='cuda'):
    """
    Apply fusion strategy - PCA-512 with whitening only
    MEMORY OPTIMIZED: All operations on CPU until final GPU transfer
    """
    layers = config['layers']
    fusion = config['fusion']

    logger.info(f"🔗 Fusing {layers} via {fusion} (PCA: 512, whiten: True)")
    logger.info(f"⚠️ Processing on CPU to avoid GPU OOM")

    # Single layer case (C5 only)
    if len(layers) == 1:
        feat = all_layer_features[layers[0]]

        # CRITICAL: Keep on CPU
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        else:
            feat = feat.cpu()

        if not validate_features(feat, f"raw single layer {layers[0]}"):
            logger.warning("⚠️ Validation failed, continuing anyway...")

        # L2 normalize on CPU
        normalized = l2_normalize(feat)
        validate_features(normalized, f"L2 normalized {layers[0]}")

        # Delete original features
        del feat, all_layer_features
        gc.collect()

        # Apply PCA-512 with whitening (processes on CPU, returns on GPU)
        normalized, pca_model = apply_pca_512_whiten(
            normalized, device=device, layer_info=f"single_{layers[0]}"
        )

        # Final L2 normalization (on GPU after PCA)
        final = l2_normalize(normalized)
        validate_features(final, f"final normalized {layers[0]}")

        return final, normalized.clone(), {
            'fusion_applied': 'none',
            'layers_used': layers,
            'layer_weights_applied': False,
            'pca_model': pca_model,
            'pca_dim': 512,
            'pca_whiten': True
        }

    # Multi-layer fusion (C4+C5)
    if layers != ["C4", "C5"]:
        raise ValueError("Only ['C4','C5'] fusion supported")

    if fusion == 'none':
        raise ValueError("Multi-layer requires fusion method")

    # CRITICAL: Process everything on CPU
    logger.info("Processing C4 and C5 on CPU...")
    processed = {}
    for layer in layers:
        feat = all_layer_features[layer]

        # Ensure on CPU
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        else:
            feat = feat.cpu()

        if not validate_features(feat, f"raw {layer}"):
            logger.warning(f"⚠️ Validation failed for {layer}, continuing anyway...")

        # L2 normalize on CPU
        processed[layer] = l2_normalize(feat)
        validate_features(processed[layer], f"L2 normalized {layer} (pre-fusion)")

    c4, c5 = processed["C4"], processed["C5"]

    # Delete source data to free memory
    del all_layer_features, processed
    gc.collect()
    log_system_memory()

    # Apply max fusion on CPU
    logger.info("Applying fusion on CPU...")
    if c4.shape[1] != c5.shape[1]:
        logger.info(f"Projecting C4 from {c4.shape[1]} to {c5.shape[1]} dims")
        # Use CPU for projection
        W = torch.randn(c5.shape[1], c4.shape[1]) * 0.01
        c4_proj = F.linear(c4, W)
        del W
        gc.collect()
    else:
        c4_proj = c4

    fused = torch.max(c4_proj, c5)
    logger.info(f"Element-wise max fusion (CPU)")

    # Delete intermediate tensors
    del c4, c5, c4_proj
    gc.collect()
    log_system_memory()

    if not validate_features(fused, "fused (before PCA)"):
        logger.warning("⚠️ Validation failed, continuing anyway...")

    logger.info(f"Fused features on CPU: {tuple(fused.shape)}")

    # Apply PCA-512 with whitening (processes on CPU, returns on GPU)
    fused, pca_model = apply_pca_512_whiten(
        fused, device=device, layer_info=f"{fusion}_{'_'.join(layers)}"
    )

    # Final L2 normalization (now on GPU after PCA)
    fused = l2_normalize(fused)

    if not validate_features(fused, "final fused & normalized"):
        logger.warning("⚠️ Final validation failed, continuing anyway...")

    logger.info(f"✅ Fusion complete: {tuple(fused.shape)} on {fused.device}")

    return fused, fused.clone(), {
        'fusion_applied': fusion,
        'layers_used': layers,
        'layer_weights_applied': True,
        'pca_model': pca_model,
        'pca_dim': 512,
        'pca_whiten': True
    }

# ============================================================================
# C1 PREPROCESSING
# ============================================================================
def preprocess_c1(path: str) -> Image.Image:
    """C1 preprocessing: grayscale→RGB, rotation correction, light denoising"""
    try:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 15)

        return Image.fromarray(img)
    except Exception as e:
        logger.warning(f"⚠️ C1 preprocess failed for {path}: {e}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as e2:
            logger.error(f"❌ Basic preprocessing failed for {path}: {e2}")
            raise e2

COMMON_TRANSFORM = transforms.Compose([
    transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================================
# DATASET
# ============================================================================
class RetrievalDataset(Dataset):
    def __init__(self, paths, preprocess_fn):
        self.paths = paths
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = COMMON_TRANSFORM(self.preprocess_fn(p))
            return img, os.path.basename(p)
        except Exception as e:
            logger.warning(f"Dataset item failed for {p}: {e}")
            return None, os.path.basename(p)

def collate_fn(batch):
    valid = [(x, y) for x, y in batch if x is not None]
    if not valid:
        return torch.empty(0), []
    xs, ys = zip(*valid)
    return torch.stack(xs), list(ys)

# ============================================================================
# PROCESSING - MEMORY OPTIMIZED
# ============================================================================
def process_configuration(config_name, config, model, db_files, overwrite=False):
    """Process a single configuration for 660K dataset - MEMORY OPTIMIZED"""

    save_dir = os.path.join(FEATURES_BASE, config_name, "660K")
    os.makedirs(save_dir, exist_ok=True)

    features_l2_file = os.path.join(save_dir, "features_l2.pt")
    index_file = os.path.join(save_dir, "index.json")
    pca_file = os.path.join(save_dir, "pca_model.pkl")

    # Check if already exists
    if not overwrite and os.path.exists(features_l2_file) and os.path.exists(index_file):
        logger.info(f"✅ Features already exist for {config_name}/660K, skipping")
        return

    layers = config['layers']
    pooling = config['pooling']

    # Create dataset and dataloader
    ds = RetrievalDataset(db_files, preprocess_c1)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate_fn,
                    pin_memory=True, persistent_workers=True)

    logger.info(f"660K: extracting {len(db_files):,} images with {config_name}...")
    log_gpu_memory()
    log_system_memory()

    # Extract features
    all_layer_features = {layer: [] for layer in layers}
    all_stems = []

    with torch.no_grad():
        for bidx, (imgs, fnames) in enumerate(tqdm(dl, desc=f"660K {config_name}")):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE, non_blocking=True)
            feats_dict = model(imgs)

            for layer in layers:
                if layer not in feats_dict:
                    logger.error(f"❌ Missing {layer} in model output")
                    return

                pooled = pool_features(feats_dict[layer], pooling=pooling, device=DEVICE)

                # CRITICAL FIX: Store on CPU immediately to save GPU memory
                all_layer_features[layer].append(pooled.cpu())

            all_stems.extend([os.path.splitext(f)[0] for f in fnames])

                    # Frequent cache clearing for 660K
            if bidx % 50 == 0:
                clear_gpu_cache()

            # Log memory every 200 batches
            if bidx % 200 == 0 and bidx > 0:
                log_system_memory()

    # MEMORY-EFFICIENT CONCATENATION
    logger.info("📊 Concatenating batches (memory-optimized)...")
    log_system_memory()

    for layer in layers:
        n_batches = len(all_layer_features[layer])

        # Check available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        # Estimate memory needed for concatenation
        if n_batches > 0:
            sample = all_layer_features[layer][0]
            sample_size = sample.element_size() * sample.nelement()
            estimated_gb = (sample_size * n_batches) / (1024**3)

            logger.info(f"{layer}: {n_batches} batches, ~{estimated_gb:.2f}GB needed, {available_gb:.2f}GB available")

            # Use chunked concatenation if memory is tight
            if estimated_gb > available_gb * 0.6:
                logger.info(f"⚠️ Using chunked concatenation for {layer}")

                # Calculate safe chunk size
                chunk_size = max(1, int(n_batches * available_gb * 0.3 / estimated_gb))
                logger.info(f"Chunk size: {chunk_size} batches")

                chunks = []
                for i in range(0, n_batches, chunk_size):
                    end = min(i + chunk_size, n_batches)
                    logger.info(f"  Concatenating chunk {i//chunk_size + 1}: batches {i}-{end}")

                    chunk = torch.cat(all_layer_features[layer][i:end], dim=0)
                    chunks.append(chunk)

                    # Clear processed batches to free memory
                    for j in range(i, end):
                        all_layer_features[layer][j] = None

                    # Force garbage collection
                    gc.collect()

                logger.info(f"  Final concatenation of {len(chunks)} chunks...")
                all_layer_features[layer] = torch.cat(chunks, dim=0)
                del chunks
                gc.collect()

            else:
                # Direct concatenation (already on CPU, no .cpu() needed)
                all_layer_features[layer] = torch.cat(all_layer_features[layer], dim=0)

        logger.info(f"{layer} final: {tuple(all_layer_features[layer].shape)}")
        log_system_memory()

    # Apply fusion with PCA-512 whitening
    logger.info("🔄 Applying fusion with PCA-512 whitening...")
    try:
        final_features, _, fusion_meta = apply_fusion(all_layer_features, config, device=DEVICE)
        if final_features is None:
            logger.error("❌ Fusion failed")
            return
        logger.info(f"✅ Final features: {tuple(final_features.shape)} on {final_features.device}")
    except Exception as e:
        logger.error(f"❌ Fusion error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save L2 normalized features
    F_l2 = final_features.cpu()
    validate_features(F_l2, "L2 normalized")

    # Create index mapping
    index_map = {stem: i for i, stem in enumerate(all_stems)}

    # Save features
    logger.info(f"💾 Saving features to {save_dir}")
    torch.save(F_l2, features_l2_file)
    logger.info(f"   - L2:  {features_l2_file}")

    # Save PCA model
    if fusion_meta.get('pca_model') is not None:
        logger.info(f"💾 Saving PCA model to {pca_file}")
        with open(pca_file, 'wb') as f:
            pickle.dump(fusion_meta['pca_model'], f)

    # Save metadata
    meta = {
        "index_mapping": index_map,
        "feature_shape": list(F_l2.shape),
        "total_images": len(all_stems),
        "preprocessing": "c1",
        "method": "dino_fusion_pca",
        "backbone": "dino_resnet50",
        "layers": layers,
        "pooling": pooling,
        "fusion": fusion_meta.get('fusion_applied', 'none'),
        "pca_applied": True,
        "pca_dim": 512,
        "pca_whiten": True,
        "db_size": "660K",
        "normalizations": ["l2"],
        "normalization_strategy": "L2 before fusion, PCA-512 with whitening on CPU, final L2",
        "layer_weights_applied": fusion_meta.get('layer_weights_applied', False),
        "version": "v4.2_660k_pca512_whiten_chunked_concat"
    }

    with open(index_file, "w") as f:
        json.dump(meta, f, indent=2)

    log_gpu_memory()
    log_system_memory()
    clear_gpu_cache()

# ============================================================================
# MAIN
# ============================================================================
def main():
    global DATA_DIR, FEATURES_BASE, BATCH_SIZE, NUM_WORKERS, DEVICE
    parser = argparse.ArgumentParser(
        description="DINO Feature Extraction - 660K Dataset - PCA-512 with Whitening ONLY (Memory Optimized)"
    )
    parser.add_argument("--overwrite", action="store_true", default=False,
                       help="Overwrite existing features")
    parser.add_argument("--data-dir", default=DATA_DIR,
                       help="Directory containing crop images (or DINO_DATA_DIR)")
    parser.add_argument("--features-base", default=FEATURES_BASE,
                       help="Output directory for feature indexes (or DINO_FEATURES_BASE)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                       help="Inference device; auto selects CUDA when available")
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    FEATURES_BASE = os.path.abspath(args.features_base)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available")
    os.makedirs(FEATURES_BASE, exist_ok=True)

    logger.info(f"🚀 Starting DINO Feature Extraction - 660K Dataset")
    logger.info(f"PCA: 512 dimensions with whitening ONLY (CPU-optimized)")
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    log_system_memory()

    # Check data directory
    if not os.path.exists(DATA_DIR):
        logger.error(f"❌ Data directory not found: {DATA_DIR}")
        return

    # Load database files
    files = sorted(set(sum([glob(os.path.join(DATA_DIR, "**", ext), recursive=True) for ext in IMG_EXTS], [])))
    if not files:
        logger.error(f"❌ No images found in {DATA_DIR}")
        return

    logger.info(f"660K: found {len(files):,} images")

    # Load model
    logger.info("🤖 Loading DINO ResNet-50 with C4/C5 hooks...")
    model = HookBasedDINO(DEVICE).eval()
    log_gpu_memory()

    # Process configurations
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING 2 CONFIGURATIONS (PCA-512 with whitening)")
    logger.info(f"{'='*80}\n")

    processed = failed = 0

    for i, (config_name, cfg) in enumerate(CONFIGS.items(), 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"[{i}/2] Processing: {config_name}")
        logger.info(f"{'─'*80}")

        try:
            process_configuration(config_name, cfg, model, files, args.overwrite)
            processed += 1
            logger.info(f"✅ COMPLETED: {config_name}")
        except Exception as e:
            logger.error(f"❌ Failed {config_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            clear_gpu_cache()

    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Processed: {processed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"💾 Features saved to: {FEATURES_BASE}")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
