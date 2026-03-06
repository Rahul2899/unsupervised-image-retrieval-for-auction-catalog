#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEFAULT_DB_IMAGE_DIR = os.environ.get("DINO_DB_IMAGE_DIR", "<HPC_DATA_ROOT>/<HPC_ACCOUNT>/preparation/total_new_crops")
DEFAULT_FEATURES_BASE = os.environ.get(
    "DINO_FEATURES_BASE",
    os.path.join(os.getcwd(), "features_dino_multifusion_multipooling"),
)
DEFAULT_HOST = os.environ.get("DINO_APP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("DINO_APP_PORT", "7865"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DIR_RE = re.compile(r"_dir\d+$", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dino_retrieval_app")

DINO_CONFIGS = {
    "C4+C5 Fusion (MAC, PCA-512)": "C4_C5_max_mac_c1_pca512_whiten/660K",
    "C5 Only (MAC, PCA-512)": "C5_mac_c1_pca512_whiten/660K",
}


@dataclass
class AppState:
    db_image_dir: str
    features_base: str
    image_path_cache: Dict[str, str]
    base_path_cache: Dict[str, str]
    model: Optional[nn.Module] = None
    feature_bank: Optional[np.ndarray] = None
    feature_index: Optional[faiss.IndexFlatIP] = None
    filenames: Optional[List[str]] = None
    config_info: Optional[dict] = None
    current_config: Optional[str] = None
    projection_w: Optional[torch.Tensor] = None


class HookBasedDINO(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("Loading DINO ResNet-50 backbone...")
        model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        self.backbone = model.to(DEVICE).eval()
        self.features = {}

        def hook(name: str):
            def fn(_, __, out):
                self.features[name] = out.detach()

            return fn

        self.backbone.layer3.register_forward_hook(hook("C4"))
        self.backbone.layer4.register_forward_hook(hook("C5"))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.features = {}
        _ = self.backbone(x.to(DEVICE))
        return self.features


def _rotate_for_display(img: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 1e-6:
        return img
    return img.convert("RGB").rotate(-angle, expand=True, fillcolor=(0, 0, 0))


def preprocess_c1_steps(img: Image.Image) -> Tuple[Image.Image, List[Tuple[Image.Image, str]]]:
    steps: List[Tuple[Image.Image, str]] = []
    rgb = img.convert("RGB")
    steps.append((rgb, "1. Original RGB"))

    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    steps.append((Image.fromarray(gray_rgb), "2. Grayscale (3-channel)"))

    h, w = gray_rgb.shape[:2]
    orientation_applied = h > w
    oriented = cv2.rotate(gray_rgb, cv2.ROTATE_90_CLOCKWISE) if orientation_applied else gray_rgb
    orient_label = "3. Orientation Fix (90° CW)" if orientation_applied else "3. Orientation Fix (no change)"
    steps.append((Image.fromarray(oriented), orient_label))

    denoised = cv2.fastNlMeansDenoisingColored(oriented, None, 3, 3, 7, 15)
    denoised_img = Image.fromarray(denoised)
    steps.append((denoised_img, "4. Denoised"))

    model_input = transforms.CenterCrop(IMAGE_SIZE)(
        transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC)(denoised_img)
    )
    steps.append((model_input, f"5. Model Input Preview ({IMAGE_SIZE}x{IMAGE_SIZE})"))

    return denoised_img, steps


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def pool_mac(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_max_pool2d(x, (1, 1)).flatten(1) + 1e-8


def build_image_cache(state: AppState) -> None:
    logger.info("Building image cache from %s", state.db_image_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    state.image_path_cache.clear()
    state.base_path_cache.clear()

    for root, _, files in os.walk(state.db_image_dir):
        for filename in files:
            if not filename.lower().endswith(exts):
                continue
            stem = os.path.splitext(filename)[0]
            path = os.path.join(root, filename)
            state.image_path_cache[stem] = path
            base = DIR_RE.sub("", stem)
            state.base_path_cache.setdefault(base, path)

    logger.info("Cached %s images", f"{len(state.image_path_cache):,}")


def find_image_path(state: AppState, index_stem: str) -> Optional[str]:
    if index_stem in state.image_path_cache:
        return state.image_path_cache[index_stem]

    base = DIR_RE.sub("", index_stem)
    if base in state.base_path_cache:
        return state.base_path_cache[base]

    return None


def get_projection_w(state: AppState, in_dim: int, out_dim: int) -> torch.Tensor:
    if state.projection_w is not None and state.projection_w.shape == (out_dim, in_dim):
        return state.projection_w

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(42)
    state.projection_w = torch.randn(out_dim, in_dim, generator=gen, device=DEVICE) * 0.01
    return state.projection_w


@torch.no_grad()
def encode_query(state: AppState, img: Optional[Image.Image] = None, preprocessed: Optional[Image.Image] = None) -> np.ndarray:
    if preprocessed is None:
        if img is None:
            raise ValueError("Either img or preprocessed must be provided.")
        preprocessed, _ = preprocess_c1_steps(img)
    x = TRANSFORM(preprocessed).unsqueeze(0).to(DEVICE)
    feats = state.model(x)

    pooled = {}
    for layer in state.config_info["layers"]:
        pooled[layer] = F.normalize(pool_mac(feats[layer]), p=2, dim=1)

    if len(pooled) == 1:
        final = next(iter(pooled.values()))
    else:
        c4 = pooled["C4"]
        c5 = pooled["C5"]
        if c4.shape[1] != c5.shape[1]:
            c4 = F.linear(c4, get_projection_w(state, c4.shape[1], c5.shape[1]))
        final = F.normalize(torch.max(c4, c5), p=2, dim=1)

    vec = final.cpu().numpy().astype(np.float32)

    if state.config_info.get("pca") is not None:
        vec = state.config_info["pca"].transform(vec)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12

    return vec


def load_features(state: AppState, config_label: str) -> None:
    config_rel_path = DINO_CONFIGS[config_label]
    config_path = os.path.join(state.features_base, config_rel_path)
    features_path = os.path.join(config_path, "features_l2.pt")
    index_path = os.path.join(config_path, "index.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")

    feats = torch.load(features_path, map_location="cpu").numpy().astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12

    with open(index_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mapping = metadata["index_mapping"]
    filenames = [None] * len(mapping)
    for stem, idx in mapping.items():
        filenames[idx] = stem

    pca_model = None
    pca_path = os.path.join(config_path, "pca_model.pkl")
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca_model = pickle.load(f)

    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)

    state.feature_bank = feats
    state.feature_index = index
    state.filenames = filenames
    state.config_info = {"layers": metadata["layers"], "pca": pca_model}
    state.current_config = config_label

    logger.info(
        "Loaded %s | vectors=%s | dim=%s",
        config_label,
        f"{feats.shape[0]:,}",
        feats.shape[1],
    )


def search(
    state: AppState,
    img: Image.Image,
    top_k: int,
    config_label: str,
    result_rotation_deg: float,
) -> Tuple[List[Tuple[Image.Image, str]], str, List[Tuple[Image.Image, str]]]:
    if img is None:
        return [], "Upload a query image to search.", []

    if state.current_config != config_label:
        load_features(state, config_label)

    top_k = int(top_k)
    preprocessed_for_query, preprocessing_steps = preprocess_c1_steps(img)
    preprocess_gallery = [(_rotate_for_display(step_img, result_rotation_deg), label) for step_img, label in preprocessing_steps]

    query = encode_query(state, preprocessed=preprocessed_for_query)
    scores, indices = state.feature_index.search(query, top_k)

    results: List[Tuple[Image.Image, str]] = []
    missing: List[str] = []

    for rank, db_idx in enumerate(indices[0], start=1):
        stem = state.filenames[db_idx]
        resolved = find_image_path(state, stem)
        if resolved and os.path.exists(resolved):
            db_img = Image.open(resolved).convert("RGB")
            db_img = _rotate_for_display(db_img, result_rotation_deg)
            results.append((db_img, f"Rank {rank} | {stem}"))
        else:
            missing.append(stem)

    status_lines = [
        f"Configuration: {config_label}",
        f"Device: {DEVICE}",
        f"Display rotation: {result_rotation_deg:.0f}°",
        f"Preprocessing steps shown: {len(preprocessing_steps)}",
        f"Returned {len(results)}/{top_k} matches",
    ]
    if missing:
        status_lines.append("Missing paths: " + ", ".join(missing[:6]))

    return results, "\n".join(status_lines), preprocess_gallery


def build_ui(state: AppState) -> gr.Blocks:
    css = """
    .app-shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 10px 8px 20px;
    }
    .hero {
      background: linear-gradient(135deg, #0f172a, #164e63 55%, #14532d);
      color: #f8fafc;
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 14px;
    }
    .hero h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.2px;
    }
    .hero p {
      margin: 8px 0 0;
      opacity: 0.92;
      font-size: 14px;
    }
    #status-box textarea {
      font-family: Menlo, Consolas, monospace;
      font-size: 12px;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default(), title="DINO Retrieval") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
                <div class='hero'>
                    <h1>DINO Retrieval Explorer</h1>
                    <p>Search visually similar auction images from a 660K feature bank using DINO + PCA-512 embeddings.</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Query Image", height=320)
                    config_input = gr.Dropdown(
                        choices=list(DINO_CONFIGS.keys()),
                        value=list(DINO_CONFIGS.keys())[0],
                        label="Embedding Configuration",
                    )
                    k_input = gr.Slider(minimum=1, maximum=60, value=12, step=1, label="Top-K Results")
                    rotation_input = gr.Slider(
                        minimum=-180,
                        maximum=180,
                        value=0,
                        step=5,
                        label="Display Rotation (degrees)",
                    )
                    with gr.Row():
                        search_btn = gr.Button("Run Search", variant="primary")
                        clear_btn = gr.ClearButton(components=[image_input], value="Clear Image")
                    status_output = gr.Textbox(label="Run Status", lines=6, elem_id="status-box")
                    preprocess_output = gr.Gallery(
                        label="Preprocessing View (Step by Step)",
                        columns=[2],
                        object_fit="contain",
                        height=340,
                    )

                with gr.Column(scale=2):
                    gallery_output = gr.Gallery(
                        label="Retrieved Results",
                        columns=[4],
                        object_fit="contain",
                        height=620,
                    )

            search_btn.click(
                fn=lambda img, k, cfg, rot: search(state, img, k, cfg, rot),
                inputs=[image_input, k_input, config_input, rotation_input],
                outputs=[gallery_output, status_output, preprocess_output],
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINO image retrieval app")
    parser.add_argument("--db-image-dir", default=DEFAULT_DB_IMAGE_DIR, help="Root directory of searchable images")
    parser.add_argument("--features-base", default=DEFAULT_FEATURES_BASE, help="Base directory containing extracted features")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Gradio host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Gradio port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = AppState(
        db_image_dir=args.db_image_dir,
        features_base=args.features_base,
        image_path_cache={},
        base_path_cache={},
    )

    logger.info("Starting DINO retrieval app on %s", DEVICE)
    if not os.path.isdir(state.db_image_dir):
        raise FileNotFoundError(f"Database image directory not found: {state.db_image_dir}")
    if not os.path.isdir(state.features_base):
        raise FileNotFoundError(f"Features directory not found: {state.features_base}")

    build_image_cache(state)
    state.model = HookBasedDINO()
    load_features(state, list(DINO_CONFIGS.keys())[0])

    demo = build_ui(state)
    demo.launch(server_name=args.host, server_port=args.port, allowed_paths=[state.db_image_dir])


if __name__ == "__main__":
    main()
