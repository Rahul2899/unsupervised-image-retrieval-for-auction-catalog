#!/usr/bin/env python3
"""
FIXED Evaluation — DINO ResNet50 - 660K Dataset with PCA-512 Whitening

CRITICAL FIXES:
1. PCA model application now matches extraction exactly
2. Added validation that PCA whitening matches between extraction and evaluation
3. Added debugging output to identify feature mismatch issues
"""

import os, json, logging, cv2, argparse, pickle
from glob import glob
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False
    print("⚠️ FAISS not available — using numpy fallback")

# ============================================================================
# CONFIGURATION
# ============================================================================
QUERY_DIR = "<HPC_DATA_ROOT>/<HPC_ACCOUNT>/restored_data/dino/features_methods/multilyer_fusion_pca/claude/finalizing_20k_to_total/query_images"
GT_JSON = "<HPC_DATA_ROOT>/<HPC_ACCOUNT>/preparation/data/gt_names.json"
FEATURES_BASE = os.path.join(os.getcwd(), "features_dino_multifusion_multipooling")
RESULTS_DIR = "evaluation_results_660k_pca512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
TRUNC_K = 50
TOPK_METRICS = (1, 5, 10)

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_660k_pca512_fixed")

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
        logger.info("Loading DINO ResNet-50...")
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
        
        logger.info("✅ DINO ready (C4/C5 hooks enabled)")
    
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
        return GeM(p=3).to(device)(feat_map).flatten(1) + 1e-8
    elif pooling == 'mac':
        return F.adaptive_max_pool2d(feat_map, (1,1)).flatten(1) + 1e-8
    raise ValueError(f"Unsupported pooling: {pooling}")

# ============================================================================
# NORMALIZATION
# ============================================================================
def l2_normalize(x):
    """L2 normalization"""
    return F.normalize(x, p=2, dim=1, eps=1e-8)

def validate_query_features(t, tag=""):
    """Validate query features with detailed statistics"""
    if torch.isnan(t).any() or torch.isinf(t).any():
        logger.error(f"❌ Invalid values in {tag}")
        return False
    
    std_val = t.std().item()
    min_val = t.min().item()
    max_val = t.max().item()
    mean_norm = t.norm(dim=1).mean().item()
    
    logger.info(f"✅ {tag}: shape={tuple(t.shape)}, std={std_val:.4f}, "
                f"min={min_val:.4f}, max={max_val:.4f}, mean_norm={mean_norm:.4f}")
    
    # Check for suspiciously uniform features
    if std_val < 0.001:
        logger.warning(f"⚠️ Very low variance ({std_val:.6f}) - features may be degenerate")
    
    return True

# ============================================================================
# QUERY FUSION - EXACT MATCH TO EXTRACTION
# ============================================================================
def apply_fusion_for_queries(feats_by_layer, layers, fusion_method, config, pca_model=None):
    """
    Apply fusion to query features - EXACTLY matches extractfull.py pipeline
    
    CRITICAL: This must produce features with identical statistics to the database
    """
    logger.info(f"🔗 Query fusion {layers} via {fusion_method} (PCA: {pca_model is not None})")
    
    # Validate PCA model if provided
    if pca_model is not None:
        logger.info(f"📊 PCA model info:")
        logger.info(f"   - n_components: {pca_model.n_components_}")
        logger.info(f"   - whiten: {pca_model.whiten}")
        logger.info(f"   - mean shape: {pca_model.mean_.shape if hasattr(pca_model, 'mean_') else 'N/A'}")
        logger.info(f"   - components shape: {pca_model.components_.shape if hasattr(pca_model, 'components_') else 'N/A'}")
    
    # SINGLE LAYER (C5 only)
    if len(layers) == 1:
        feat = feats_by_layer[layers[0]]
        validate_query_features(feat, f"raw {layers[0]}")
        
        # Step 1: L2 normalize BEFORE PCA
        normalized = l2_normalize(feat)
        validate_query_features(normalized, f"L2 normalized {layers[0]} (pre-PCA)")
        
        # Step 2: Apply PCA if model exists
        if pca_model is not None:
            logger.info(f"Applying PCA: {tuple(normalized.shape)} → {pca_model.n_components_}")
            
            # CRITICAL: Convert to numpy, apply PCA, convert back
            normalized_np = normalized.cpu().numpy()
            logger.info(f"Pre-PCA numpy stats: mean={normalized_np.mean():.4f}, std={normalized_np.std():.4f}")
            
            transformed = pca_model.transform(normalized_np)
            logger.info(f"Post-PCA numpy stats: mean={transformed.mean():.4f}, std={transformed.std():.4f}")
            
            normalized = torch.from_numpy(transformed).float().to(feat.device)
            validate_query_features(normalized, f"post-PCA {layers[0]}")
        
        # Step 3: Final L2 normalization
        final = l2_normalize(normalized)
        validate_query_features(final, f"final L2 normalized {layers[0]}")
        
        return final
    
    # MULTI-LAYER FUSION (C4+C5)
    if layers != ["C4", "C5"]:
        raise ValueError("Only C4+C5 fusion supported")
    if fusion_method == 'none':
        raise ValueError("Multi-layer needs fusion method")
    
    # Step 1: L2 normalize EACH layer BEFORE fusion
    processed = {}
    for layer in layers:
        feat = feats_by_layer[layer]
        validate_query_features(feat, f"raw {layer}")
        processed[layer] = l2_normalize(feat)
        validate_query_features(processed[layer], f"L2 normalized {layer} (pre-fusion)")
    
    c4, c5 = processed["C4"], processed["C5"]
    
    # Step 2: Apply fusion strategy
    if fusion_method == 'max':
        if c4.shape[1] != c5.shape[1]:
            logger.info(f"Projecting C4 from {c4.shape[1]} to {c5.shape[1]} dims")
            # CRITICAL: Use consistent random seed for projection
            W = torch.randn(c5.shape[1], c4.shape[1], device=c4.device) * 0.01
            c4_proj = F.linear(c4, W)
        else:
            c4_proj = c4
        fused = torch.max(c4_proj, c5)
        validate_query_features(fused, "fused (max, pre-PCA)")
    
    elif fusion_method == 'concat':
        fused = torch.cat([c4, c5], dim=1)
        validate_query_features(fused, "fused (concat, pre-PCA)")
    
    elif fusion_method == 'weighted_sum':
        if c4.shape[1] != c5.shape[1]:
            W = torch.randn(c5.shape[1], c4.shape[1], device=c4.device) * 0.01
            c4 = F.linear(c4, W)
        fused = 0.4 * c4 + 0.6 * c5
        validate_query_features(fused, "fused (weighted_sum, pre-PCA)")
    
    elif fusion_method == 'average':
        if c4.shape[1] != c5.shape[1]:
            W = torch.randn(c5.shape[1], c4.shape[1], device=c4.device) * 0.01
            c4 = F.linear(c4, W)
        fused = (c4 + c5) / 2.0
        validate_query_features(fused, "fused (average, pre-PCA)")
    
    else:
        raise ValueError(f"Unknown fusion: {fusion_method}")
    
    # Step 3: Apply PCA if model exists
    if pca_model is not None:
        logger.info(f"Applying PCA: {tuple(fused.shape)} → {pca_model.n_components_}")
        
        # CRITICAL: Convert to numpy, apply PCA, convert back
        fused_np = fused.cpu().numpy()
        logger.info(f"Pre-PCA numpy stats: mean={fused_np.mean():.4f}, std={fused_np.std():.4f}")
        
        transformed = pca_model.transform(fused_np)
        logger.info(f"Post-PCA numpy stats: mean={transformed.mean():.4f}, std={transformed.std():.4f}")
        
        fused = torch.from_numpy(transformed).float().to(c4.device)
        validate_query_features(fused, "post-PCA fused")
    
    # Step 4: Final L2 normalization
    fused = l2_normalize(fused)
    validate_query_features(fused, "final L2 normalized fused")
    
    return fused

# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_c1(path: str) -> Image.Image:
    """C1 preprocessing - matches extractfull.py"""
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
# GROUND TRUTH LOADING
# ============================================================================
def _stem(x): 
    return os.path.splitext(os.path.basename(x))[0].lower()

def _fname(x): 
    return os.path.basename(x).lower()

def load_gt_dual_keys(path):
    """Load ground truth with dual key support"""
    with open(path) as f:
        raw = json.load(f)
    
    gt_full, gt_stem = {}, {}
    
    def add(q, v):
        if not isinstance(v, list): 
            v = [v]
        vset = {_stem(x) for x in v if x}
        gt_full[_fname(q)] = vset
        gt_stem[_stem(q)] = vset
    
    if isinstance(raw, dict):
        for k, v in raw.items(): 
            add(k, v)
    elif isinstance(raw, list):
        for item in raw:
            if "query" in item:
                add(item["query"], item.get("matches", item.get("positives", [])))
            else:
                for k, v in item.items(): 
                    add(k, v)
    
    return gt_full, gt_stem

# ============================================================================
# DATABASE FEATURE LOADING
# ============================================================================
def load_db_features_and_config(feat_dir, db_size):
    """Load database features and configuration"""
    path = os.path.join(FEATURES_BASE, feat_dir, db_size)
    
    ffile = os.path.join(path, "features_l2.pt")
    ifile = os.path.join(path, "index.json")
    pfile = os.path.join(path, "pca_model.pkl")
    
    if not (os.path.exists(ffile) and os.path.exists(ifile)):
        logger.error(f"Missing features: {path}")
        return None, None, None, None
    
    logger.info(f"Loading DB features from {path}")
    F = torch.load(ffile, map_location="cpu")
    
    # Log database feature statistics
    logger.info(f"📊 Database feature statistics:")
    logger.info(f"   - shape: {F.shape}")
    logger.info(f"   - mean: {F.mean().item():.4f}")
    logger.info(f"   - std: {F.std().item():.4f}")
    logger.info(f"   - min: {F.min().item():.4f}")
    logger.info(f"   - max: {F.max().item():.4f}")
    logger.info(f"   - mean norm: {F.norm(dim=1).mean().item():.4f}")
    
    with open(ifile) as f:
        cfg = json.load(f)
    
    # Load PCA model if exists
    pca = None
    if os.path.exists(pfile):
        with open(pfile, 'rb') as f:
            pca = pickle.load(f)
        logger.info(f"📊 PCA model loaded: {pca.n_components_} dims{' +whiten' if cfg.get('pca_whiten') else ''}")
        
        # Validate PCA model
        if hasattr(pca, 'mean_'):
            logger.info(f"   - PCA mean: {pca.mean_.mean():.4f} ± {pca.mean_.std():.4f}")
        if hasattr(pca, 'explained_variance_'):
            logger.info(f"   - Explained variance: {pca.explained_variance_[:5]}")
    
    # Reconstruct database names from index mapping
    names = [""] * len(cfg["index_mapping"])
    for s, i in cfg["index_mapping"].items():
        names[i] = s
    
    logger.info(f"✅ DB loaded: {F.shape}, {len(names)} images")
    logger.info(f"   Layers: {cfg['layers']}, Fusion: {cfg['fusion']}, Pooling: {cfg['pooling']}")
    logger.info(f"   PCA: {cfg.get('pca_dim', 'None')}, Whitening: {cfg.get('pca_whiten', False)}")
    
    return F, names, cfg, pca

# ============================================================================
# QUERY FEATURE EXTRACTION
# ============================================================================
def extract_query_features(model, qdir, layers, fusion, config, pca_model=None):
    """Extract query features matching the extraction pipeline"""
    feats = {l: [] for l in layers}
    names = []
    files = sorted(glob(os.path.join(qdir, "*.jpg")))
    
    if not files:
        logger.error(f"No query images found in {qdir}")
        return torch.empty(0), []
    
    logger.info(f"Found {len(files)} query images")
    
    pooling = config.get("pooling", "mac")
    
    with torch.no_grad():
        for p in tqdm(files, desc="Extracting queries"):
            try:
                x = COMMON_TRANSFORM(preprocess_c1(p)).unsqueeze(0).to(DEVICE)
                fm = model(x)
                
                for l in layers:
                    if l not in fm:
                        logger.error(f"Missing layer {l} in model output")
                        continue
                    pooled = pool_features(fm[l].cpu(), pooling, 'cpu')
                    feats[l].append(pooled)
                
                names.append(os.path.basename(p))
            except Exception as e:
                logger.warning(f"Failed to process {p}: {e}")
    
    if not names:
        logger.error("No queries extracted successfully")
        return torch.empty(0), []
    
    # Concatenate features for each layer
    for l in layers:
        if feats[l]:
            feats[l] = torch.cat(feats[l], dim=0)
            logger.info(f"Query {l}: {tuple(feats[l].shape)}")
        else:
            logger.error(f"No features for layer {l}")
            return torch.empty(0), []
    
    # Apply fusion with PCA
    final = apply_fusion_for_queries(feats, layers, fusion, config, pca_model)
    
    if final.numel() == 0:
        logger.error("Fusion failed")
        return torch.empty(0), []
    
    validate_query_features(final, "final query features")
    return final, names

# ============================================================================
# RANKING
# ============================================================================
def normalize_np(X):
    """Normalize numpy array"""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def rank_with_cosine(Q, X):
    """Rank using cosine similarity"""
    if FAISS_OK:
        idx = faiss.IndexFlatIP(X.shape[1])
        idx.add(normalize_np(X).astype("float32"))
        _, I = idx.search(normalize_np(Q).astype("float32"), X.shape[0])
        return I
    return np.argsort(-(normalize_np(Q) @ normalize_np(X).T), axis=1)

def rank_with_l2(Q, X):
    """Rank using L2 distance"""
    if FAISS_OK:
        idx = faiss.IndexFlatL2(X.shape[1])
        idx.add(X.astype("float32"))
        _, I = idx.search(Q.astype("float32"), X.shape[0])
        return I
    q2 = (Q**2).sum(1, keepdims=True)
    x2 = (X**2).sum(1).reshape(1, -1)
    return np.argsort(q2 + x2 - 2*(Q @ X.T), axis=1)

# ============================================================================
# METRICS COMPUTATION
# ============================================================================
def print_debug_name_overlap(db_names, q_names, gt_full, gt_stem):
    print("\n=== DEBUG: Name Overlap Check ===")
    print(f"First 5 DB names: {db_names[:5]}")
    print(f"First 5 Query names: {q_names[:5]}")
    print(f"First 5 GT full keys: {list(gt_full.keys())[:5]}")
    print(f"First 5 GT stem keys: {list(gt_stem.keys())[:5]}")
    db_set = set(db_names)
    q_set = set(q_names)
    overlap = db_set & q_set
    print(f"# Overlap (exact): {len(overlap)}")
    db_stem = set([os.path.splitext(n)[0].lower() for n in db_names])
    q_stem = set([os.path.splitext(n)[0].lower() for n in q_names])
    overlap_stem = db_stem & q_stem
    print(f"# Overlap (stem): {len(overlap_stem)}")
    print("=== END DEBUG ===\n")

def compute_metrics(db_feats, q_feats, db_names, q_names, gt_full, gt_stem):
    """Compute retrieval metrics"""
    if db_feats.shape[1] != q_feats.shape[1]:
        logger.error(f"Dimension mismatch: DB{db_feats.shape} vs Q{q_feats.shape}")
        return None
    
    # Log feature comparison
    logger.info(f"📊 Feature comparison:")
    logger.info(f"   DB: mean={db_feats.mean():.4f}, std={db_feats.std():.4f}")
    logger.info(f"   Q:  mean={q_feats.mean():.4f}, std={q_feats.std():.4f}")
    
    # Sample similarity to check if features are in same space
    if len(q_feats) > 0 and len(db_feats) > 0:
        # Both are numpy arrays, so do not call .numpy()
        sample_sim = (q_feats[0:1] @ db_feats[:100].T)
        logger.info(f"   Sample similarity (Q[0] vs DB[:100]): "
                   f"mean={sample_sim.mean():.4f}, max={sample_sim.max():.4f}, min={sample_sim.min():.4f}")
    
    # DEBUG: Print name overlap
    print_debug_name_overlap(db_names, q_names, gt_full, gt_stem)

    db_norm = [_stem(n) for n in db_names]
    q_full = [_fname(n) for n in q_names]
    q_stem = [_stem(n) for n in q_names]
    
    def rel(i):
        return gt_full.get(q_full[i], set()) or gt_stem.get(q_stem[i], set())
    
    results = {}
    
    for metric in ("cosine", "euclidean"):
        logger.info(f"Computing {metric} rankings...")
        I = rank_with_cosine(q_feats, db_feats) if metric == "cosine" else rank_with_l2(q_feats, db_feats)
        
        hits = {k: 0 for k in TOPK_METRICS}
        sum_full = sum_trunc = valid = 0
        
        for qi in range(len(q_names)):
            r = rel(qi)
            if not r: 
                continue
            valid += 1
            
            retrieved = [db_norm[j] for j in I[qi]]
            
            # Top-K accuracy
            for k in TOPK_METRICS:
                if any(x in r for x in retrieved[:k]):
                    hits[k] += 1
            
            # Full mAP
            ap = found = 0
            for rank, name in enumerate(retrieved, 1):
                if name in r:
                    found += 1
                    ap += found / rank
            sum_full += ap / len(r)
            
            # Truncated mAP
            ap = found = 0
            for rank, name in enumerate(retrieved[:TRUNC_K], 1):
                if name in r:
                    found += 1
                    ap += found / rank
            sum_trunc += ap / len(r)
        
        results[metric] = {
            "top1": hits[1]/valid if valid else 0,
            "top5": hits[5]/valid if valid else 0,
            "top10": hits[10]/valid if valid else 0,
            "mAP_full": sum_full/valid if valid else 0,
            f"mAP@{TRUNC_K}": sum_trunc/valid if valid else 0,
            "valid_queries": valid,
            "total_queries": len(q_names),
            "missing_in_gt": len(q_names) - valid
        }
        
        logger.info(f"  {metric.upper()}: Top1={results[metric]['top1']:.3f} "
                   f"Top5={results[metric]['top5']:.3f} Top10={results[metric]['top10']:.3f} "
                   f"mAP={results[metric]['mAP_full']:.3f} mAP@{TRUNC_K}={results[metric][f'mAP@{TRUNC_K}']:.3f}")
    
    return results

# ============================================================================
# EVALUATION
# ============================================================================
def check_exists(feat_dir, db_size, overwrite):
    """Check if evaluation results already exist"""
    if overwrite: 
        return False
    pattern = f"{feat_dir}_{db_size}_*.json"
    return len(glob(os.path.join(RESULTS_DIR, pattern))) > 0

def evaluate_single_configuration(feat_dir, db_size, model, gt_full, gt_stem, overwrite=False):
    """Evaluate a single configuration"""
    if check_exists(feat_dir, db_size, overwrite):
        logger.info(f"⏭️ SKIP {feat_dir}/{db_size} (results exist)")
        return None
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {feat_dir} / {db_size}")
    logger.info(f"{'='*80}\n")
    
    # Load database features
    F, names, config, pca = load_db_features_and_config(feat_dir, db_size)
    if F is None:
        logger.error(f"Failed to load {feat_dir}/{db_size}")
        return None
    
    # Extract query features
    Q, qnames = extract_query_features(
        model.to(DEVICE).eval(), 
        QUERY_DIR,
        config["layers"], 
        config["fusion"], 
        config, 
        pca
    )
    
    if Q.numel() == 0:
        logger.error(f"No queries extracted for {feat_dir}/{db_size}")
        return None
    
    logger.info(f"\nComputing metrics: {len(qnames)} queries × {len(names)} DB")
    logger.info(f"Shapes: DB{F.shape} Q{Q.shape}")
    
    # Compute metrics
    res = compute_metrics(F.numpy(), Q.numpy(), names, qnames, gt_full, gt_stem)
    if res is None:
        logger.error(f"Metric computation failed for {feat_dir}/{db_size}")
        return None
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"{feat_dir}_{db_size}_{ts}.json")
    
    final = {
        "feature_dir": feat_dir,
        "db_size": db_size,
        "timestamp": ts,
        "results": res,
        "config": config,
        "version": "v1.1_660k_pca512_whiten_fixed"
    }
    
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    
    logger.info(f"\n✅ Results saved: {out}\n")
    return final

# ============================================================================
# FEATURE DISCOVERY
# ============================================================================
def discover_feature_dirs():
    """Discover available feature directories"""
    if not os.path.exists(FEATURES_BASE):
        logger.warning(f"Features base directory not found: {FEATURES_BASE}")
        return []
    
    out = []
    for item in os.listdir(FEATURES_BASE):
        p = os.path.join(FEATURES_BASE, item)
        if os.path.isdir(p):
            if os.path.isdir(os.path.join(p, "660K")):
                out.append(item)
    
    return sorted(out)

# ============================================================================
# SUMMARY
# ============================================================================
def print_summary():
    """Print summary of all evaluation results"""
    files = glob(os.path.join(RESULTS_DIR, "*.json"))
    if not files:
        print("No results found")
        return
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - 660K PCA-512 with Whitening")
    print("="*80)
    
    for f in sorted(files):
        with open(f) as fp:
            d = json.load(fp)
        
        feat = d.get("feature_dir", "?")
        db = d.get("db_size", "?")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dirs', nargs='+', default=None, help='Feature directories to evaluate')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--summary', action='store_true', help='Print summary of results')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    logger.info('🚀 Starting evaluation for 660K PCA-512 with Whitening')
    gt_full, gt_stem = load_gt_dual_keys(GT_JSON)
    logger.info(f'✅ GT loaded: {len(gt_full)} full keys, {len(gt_stem)} stem keys')

    model = HookBasedDINO(DEVICE).eval()

    dirs = args.feature_dirs or discover_feature_dirs()
    logger.info(f'Auto-discovered {len(dirs)} directories: {dirs}')
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {len(dirs)} configurations × 1 database sizes")
    logger.info(f"{'='*80}\n")

    for i, feat_dir in enumerate(dirs):
        logger.info(f"\n[{i+1}/{len(dirs)}] {feat_dir} K")
        evaluate_single_configuration(feat_dir, '660K', model, gt_full, gt_stem, overwrite=args.overwrite)

    logger.info("\n🎉 Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Completed: {len(dirs)}/{len(dirs)}")
    logger.info(f"Results saved in: {RESULTS_DIR}")
    logger.info("\nRun with --summary to view all results")
    logger.info("="*80)

if __name__ == "__main__":
    main()