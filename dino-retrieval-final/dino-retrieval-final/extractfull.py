#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import pickle
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DATA_DIR = os.environ.get("DINO_DATA_DIR", "<HPC_DATA_ROOT>/<HPC_ACCOUNT>/preparation/total_new_crops")
FEATURES_BASE = os.environ.get(
    "DINO_FEATURES_BASE",
    os.path.join(os.getcwd(), "features_dino_multifusion_multipooling"),
)
BATCH_SIZE = int(os.environ.get("DINO_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("DINO_NUM_WORKERS", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")

CONFIGS = {
    "C4_C5_max_mac_c1_pca512_whiten": {
        "layers": ["C4", "C5"],
        "fusion": "max",
        "pooling": "mac",
        "pca_dim": 512,
        "pca_whiten": True,
    },
    "C5_mac_c1_pca512_whiten": {
        "layers": ["C5"],
        "fusion": "none",
        "pooling": "mac",
        "pca_dim": 512,
        "pca_whiten": True,
    },
}

os.makedirs(FEATURES_BASE, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dino_feature_extractor")


class HookBasedDINO(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        logger.info("Loading DINO ResNet-50")
        model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        self.backbone = model.to(device).eval()
        self.device = device
        self.features = {}

        def hook(name: str):
            def fn(_, __, out):
                self.features[name] = out.detach().clone()

            return fn

        self.backbone.layer3.register_forward_hook(hook("C4"))
        self.backbone.layer4.register_forward_hook(hook("C5"))

    def forward(self, x: torch.Tensor):
        self.features = {}
        _ = self.backbone(x.to(self.device))
        return self.features


def log_gpu_memory() -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info("GPU memory allocated=%.2f GB reserved=%.2f GB", allocated, reserved)


def log_system_memory() -> None:
    mem = psutil.virtual_memory()
    logger.info(
        "RAM used=%.2f/%.2f GB (%.1f%%), available=%.2f GB",
        mem.used / 1024**3,
        mem.total / 1024**3,
        mem.percent,
        mem.available / 1024**3,
    )


def clear_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def pool_features(feat_map: torch.Tensor, pooling: str = "mac", device: str = "cuda") -> torch.Tensor:
    if pooling == "mac":
        pooled = F.adaptive_max_pool2d(feat_map, (1, 1)).flatten(1)
    elif pooling == "gem":
        pooled = F.adaptive_avg_pool2d(feat_map.clamp(min=1e-6).pow(3.0), (1, 1)).pow(1 / 3.0).flatten(1)
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    return pooled.to(device) + 1e-8


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=1e-8)


def validate_features(tensor: torch.Tensor, tag: str = "") -> bool:
    check = tensor.cpu() if tensor.is_cuda else tensor
    if torch.isnan(check).any():
        logger.error("NaN detected: %s", tag)
        return False
    if torch.isinf(check).any():
        logger.error("Inf detected: %s", tag)
        return False

    near_zero = (check.norm(dim=1) < 1e-6).sum().item()
    if near_zero > 0:
        logger.warning("Near-zero vectors in %s: %d", tag, near_zero)

    std_val = check.std().item()
    mean_norm = check.norm(dim=1).mean().item()
    logger.info("Validated %s | shape=%s std=%.5f mean_norm=%.5f", tag, tuple(tensor.shape), std_val, mean_norm)
    return True


def apply_pca_512_whiten(features_tensor: torch.Tensor, device: str, layer_info: str):
    target_dim = 512
    logger.info("Applying PCA for %s: %s -> %d", layer_info, tuple(features_tensor.shape), target_dim)

    features_cpu = features_tensor.cpu()
    del features_tensor
    clear_gpu_cache()

    validate_features(features_cpu, f"pre_pca_{layer_info}")
    matrix = features_cpu.numpy().astype(np.float32)
    del features_cpu

    rows, cols = matrix.shape
    actual_dim = min(target_dim, min(rows, cols) - 1)
    if actual_dim != target_dim:
        logger.warning("Reducing PCA dim from %d to %d", target_dim, actual_dim)

    if rows > 500000:
        batch_size = 2048
    elif rows > 100000:
        batch_size = 1024
    else:
        batch_size = 512

    pca = IncrementalPCA(n_components=actual_dim, whiten=True, batch_size=batch_size)

    for i in tqdm(range(0, rows, batch_size), desc=f"PCA fit {layer_info}"):
        pca.partial_fit(matrix[i : i + batch_size])

    chunks = []
    for i in tqdm(range(0, rows, batch_size), desc=f"PCA transform {layer_info}"):
        chunks.append(pca.transform(matrix[i : i + batch_size]))

    projected = np.vstack(chunks)
    del chunks, matrix

    logger.info("Explained variance ratio sum=%.6f", pca.explained_variance_ratio_.sum())
    projected_tensor = torch.from_numpy(projected.astype(np.float32)).to(device)
    del projected

    validate_features(projected_tensor, f"post_pca_{layer_info}")
    return projected_tensor, pca


def apply_fusion(all_layer_features: Dict[str, torch.Tensor], config: dict, device: str):
    layers = config["layers"]
    fusion = config["fusion"]

    if len(layers) == 1:
        single = all_layer_features[layers[0]].cpu()
        validate_features(single, f"raw_{layers[0]}")
        single = l2_normalize(single)
        reduced, pca_model = apply_pca_512_whiten(single, device=device, layer_info=f"single_{layers[0]}")
        final = l2_normalize(reduced)
        return final, {
            "fusion_applied": "none",
            "layers_used": layers,
            "layer_weights_applied": False,
            "pca_model": pca_model,
        }

    if layers != ["C4", "C5"] or fusion != "max":
        raise ValueError("Only C4+C5 max fusion is supported")

    c4 = l2_normalize(all_layer_features["C4"].cpu())
    c5 = l2_normalize(all_layer_features["C5"].cpu())

    if c4.shape[1] != c5.shape[1]:
        projection = torch.randn(c5.shape[1], c4.shape[1]) * 0.01
        c4 = F.linear(c4, projection)

    fused = torch.max(c4, c5)
    reduced, pca_model = apply_pca_512_whiten(fused, device=device, layer_info="max_C4_C5")
    final = l2_normalize(reduced)
    return final, {
        "fusion_applied": fusion,
        "layers_used": layers,
        "layer_weights_applied": True,
        "pca_model": pca_model,
    }


def preprocess_c1(path: str) -> Image.Image:
    try:
        img = cv2.imread(path)
        if img is None:
            arr = np.array(Image.open(path).convert("RGB"))
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 15)
        return Image.fromarray(img)
    except Exception:
        return Image.open(path).convert("RGB")


COMMON_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class RetrievalDataset(Dataset):
    def __init__(self, paths: List[str], preprocess_fn):
        self.paths = paths
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = COMMON_TRANSFORM(self.preprocess_fn(path))
            return img, os.path.basename(path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return None, os.path.basename(path)


def collate_fn(batch):
    valid = [(x, y) for x, y in batch if x is not None]
    if not valid:
        return torch.empty(0), []
    xs, ys = zip(*valid)
    return torch.stack(xs), list(ys)


def process_configuration(config_name: str, config: dict, model: nn.Module, db_files: List[str], overwrite: bool = False):
    save_dir = os.path.join(FEATURES_BASE, config_name, "660K")
    os.makedirs(save_dir, exist_ok=True)

    features_path = os.path.join(save_dir, "features_l2.pt")
    index_path = os.path.join(save_dir, "index.json")
    pca_path = os.path.join(save_dir, "pca_model.pkl")

    if not overwrite and os.path.exists(features_path) and os.path.exists(index_path):
        logger.info("Skipping existing config: %s", config_name)
        return

    dataset = RetrievalDataset(db_files, preprocess_c1)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    all_layer_features = {layer: [] for layer in config["layers"]}
    all_stems: List[str] = []

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(loader, desc=f"Extract {config_name}")):
            if len(images) == 0:
                continue

            feats_dict = model(images.to(DEVICE, non_blocking=True))
            for layer in config["layers"]:
                pooled = pool_features(feats_dict[layer], pooling=config["pooling"], device=DEVICE)
                all_layer_features[layer].append(pooled.cpu())

            all_stems.extend(os.path.splitext(name)[0] for name in filenames)

            if batch_idx % 50 == 0:
                clear_gpu_cache()
            if batch_idx % 200 == 0 and batch_idx > 0:
                log_system_memory()

    logger.info("Concatenating extracted layer tensors")
    for layer in config["layers"]:
        all_layer_features[layer] = torch.cat(all_layer_features[layer], dim=0)
        logger.info("Layer %s shape %s", layer, tuple(all_layer_features[layer].shape))

    final_features, fusion_meta = apply_fusion(all_layer_features, config, device=DEVICE)

    cpu_features = final_features.cpu()
    validate_features(cpu_features, f"final_{config_name}")

    index_mapping = {stem: i for i, stem in enumerate(all_stems)}
    torch.save(cpu_features, features_path)

    pca_model = fusion_meta.get("pca_model")
    if pca_model is not None:
        with open(pca_path, "wb") as f:
            pickle.dump(pca_model, f)

    metadata = {
        "index_mapping": index_mapping,
        "feature_shape": list(cpu_features.shape),
        "total_images": len(all_stems),
        "preprocessing": "c1",
        "method": "dino_fusion_pca",
        "backbone": "dino_resnet50",
        "layers": fusion_meta.get("layers_used", config["layers"]),
        "pooling": config["pooling"],
        "fusion": fusion_meta.get("fusion_applied", "none"),
        "pca_applied": True,
        "pca_dim": 512,
        "pca_whiten": True,
        "db_size": "660K",
        "normalizations": ["l2"],
        "layer_weights_applied": fusion_meta.get("layer_weights_applied", False),
        "version": "v5_clean",
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    del all_layer_features, final_features, cpu_features
    gc.collect()
    clear_gpu_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINO 660K feature extractor")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to dataset image directory")
    parser.add_argument("--features-base", default=FEATURES_BASE, help="Output feature base directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    return parser.parse_args()


def main():
    global DATA_DIR, FEATURES_BASE, BATCH_SIZE, NUM_WORKERS
    args = parse_args()

    DATA_DIR = args.data_dir
    FEATURES_BASE = args.features_base
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    if DEVICE != "cuda":
        raise RuntimeError("CUDA is required for this extraction script.")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(set(sum((glob(os.path.join(DATA_DIR, ext)) for ext in IMG_EXTS), [])))
    if not files:
        raise RuntimeError(f"No images found in {DATA_DIR}")

    logger.info("Dataset files found: %s", f"{len(files):,}")
    logger.info("Output directory: %s", FEATURES_BASE)
    logger.info("Using batch_size=%d num_workers=%d", BATCH_SIZE, NUM_WORKERS)
    log_system_memory()
    log_gpu_memory()

    model = HookBasedDINO(DEVICE).eval()

    processed = 0
    failed = 0
    for name, cfg in CONFIGS.items():
        try:
            process_configuration(name, cfg, model, files, overwrite=args.overwrite)
            logger.info("Completed %s", name)
            processed += 1
        except Exception as exc:
            logger.exception("Failed %s: %s", name, exc)
            failed += 1
            clear_gpu_cache()

    logger.info("Summary processed=%d failed=%d", processed, failed)


if __name__ == "__main__":
    main()
