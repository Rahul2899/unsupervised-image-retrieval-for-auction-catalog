#!/usr/bin/env python3
import argparse, base64, io, json, os, pickle, re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from flask import send_from_directory
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR_RE = re.compile(r"_dir\d+$", re.I)
CONFIGS = {"C4+C5 Fusion": "C4_C5_max_mac_c1_pca512_whiten/660K", "C5 Only": "C5_mac_c1_pca512_whiten/660K"}
APP_ROOT = os.path.dirname(__file__)
QUERY_IMAGE_DIR = os.environ.get("DINO_QUERY_IMAGE_DIR", os.path.join(APP_ROOT, "data", "query_images"))
PDF_IMAGE_DIR = os.environ.get("DINO_PAGE_IMAGE_DIR", os.path.join(APP_ROOT, "data", "page_images"))
DEFAULT_IMAGE_DIR = os.path.join(APP_ROOT, "data", "crops_yolo_v2")
SAMPLE_QUERIES = [
    ("achenbach.jpg", "Study 01"), ("liebermann1.jpg", "Study 02"), ("truebner2.jpg", "Study 03"),
    ("liebermann3.jpg", "Study 04"), ("lier.jpg", "Study 05"), ("spitzweg.jpg", "Study 06"),
    ("liebermann2.jpg", "Study 07"), ("richter.jpg", "Study 08"), ("buerkel.jpg", "Study 09"),
    ("zumbusch.jpg", "Study 10"), ("hagemeister.jpg", "Study 11"), ("truebner.jpg", "Study 12"),
]
TRANSFORM = transforms.Compose([transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

@dataclass
class RetrievalState:
    image_dir: str; features_dir: str; paths: dict; base_paths: dict
    provenance: dict
    model: Optional[nn.Module] = None; index: Optional[object] = None
    stems: Optional[list] = None; config: Optional[dict] = None; method: Optional[str] = None; projection: Optional[torch.Tensor] = None
    indexes: dict = field(default_factory=dict)
    stems_by_method: dict = field(default_factory=dict)
    configs_by_method: dict = field(default_factory=dict)
    projections: dict = field(default_factory=dict)

class ExactInnerProductIndex:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query, limit):
        scores = self.vectors @ query[0]
        positions = np.argpartition(scores, -limit)[-limit:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        return scores[positions][None], positions[None]

class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        self.backbone, self.features = model.to(DEVICE).eval(), {}
        self.backbone.layer3.register_forward_hook(lambda _, __, x: self.features.__setitem__("C4", x.detach()))
        self.backbone.layer4.register_forward_hook(lambda _, __, x: self.features.__setitem__("C5", x.detach()))
    def forward(self, x):
        self.features = {}; self.backbone(x.to(DEVICE)); return self.features

def preprocess(image):
    image = image.convert("L").convert("RGB")
    if image.height > image.width: image = image.transpose(Image.Transpose.ROTATE_270)
    return image

def cache_images(state):
    for root, _, files in os.walk(state.image_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                stem, path = os.path.splitext(filename)[0], os.path.join(root, filename)
                state.paths[stem] = path; state.base_paths.setdefault(DIR_RE.sub("", stem), path)
    metadata_path = os.environ.get("DINO_METADATA_FILE", os.path.join(state.image_dir, "crop_metadata.jsonl"))
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    stem = os.path.splitext(os.path.basename(record.get("crop_path", "")))[0]
                    if stem: state.provenance[stem] = record
                except (json.JSONDecodeError, OSError):
                    continue

def load_features(state, method):
    if method in state.indexes: return
    folder = os.path.join(state.features_dir, CONFIGS[method])
    vectors = torch.load(os.path.join(folder, "features_l2.pt"), map_location="cpu").numpy()
    # Avoid a second full-corpus copy while retaining the exact-search vectors.
    if vectors.dtype != np.float32: vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    with open(os.path.join(folder, "index.json"), encoding="utf-8") as file: metadata = json.load(file)
    stems = [None] * len(metadata["index_mapping"])
    for stem, position in metadata["index_mapping"].items(): stems[position] = stem
    pca_path = os.path.join(folder, "pca_model.pkl")
    index = ExactInnerProductIndex(vectors)
    state.indexes[method] = index
    state.stems_by_method[method] = stems
    state.configs_by_method[method] = {"layers": metadata["layers"], "pca": pickle.load(open(pca_path, "rb")) if os.path.exists(pca_path) else None}

@torch.no_grad()
def encode(state, image, config, method):
    features = state.model(TRANSFORM(preprocess(image)).unsqueeze(0).to(DEVICE))
    pooled = {layer: F.normalize(F.adaptive_max_pool2d(features[layer], (1, 1)).flatten(1) + 1e-8, p=2, dim=1) for layer in config["layers"]}
    if len(pooled) == 1: vector = next(iter(pooled.values()))
    else:
        c4, c5 = pooled["C4"], pooled["C5"]
        if c4.shape[1] != c5.shape[1]:
            if method not in state.projections: state.projections[method] = torch.randn(c5.shape[1], c4.shape[1], generator=torch.Generator(device=DEVICE).manual_seed(42), device=DEVICE) * .01
            c4 = F.linear(c4, state.projections[method])
        vector = F.normalize(torch.maximum(c4, c5), p=2, dim=1)
    vector = vector.cpu().numpy().astype(np.float32)
    if config["pca"] is not None:
        vector = config["pca"].transform(vector); vector /= np.linalg.norm(vector, axis=1, keepdims=True) + 1e-12
    return vector

@lru_cache(maxsize=1)
def get_state():
    image_dir = os.environ.get("DINO_DB_IMAGE_DIR", DEFAULT_IMAGE_DIR)
    if not os.path.isdir(image_dir): raise FileNotFoundError("DINO_DB_IMAGE_DIR must point to the searchable crop directory.")
    state = RetrievalState(image_dir, os.environ.get("DINO_FEATURES_BASE", os.path.join(APP_ROOT, "data", "features_yolo_v2")), {}, {}, {})
    cache_images(state); state.model = DINO(); return state

def provenance_for(state, stem):
    record = state.provenance.get(stem) or state.provenance.get(DIR_RE.sub("", stem)) or {}
    clean_stem = DIR_RE.sub("", stem)
    match = re.match(r"(.+)_page_?(\d+)(?:_\d+)?$", clean_stem)
    canonical = re.match(r"(.+)__pdf-(\d+)(?:__crop-\d+)?$", clean_stem)
    parsed = canonical or match
    catalogue_id = record.get("catalogue_id") or (parsed.group(1) if parsed else None)
    page_index = record.get("pdf_page_index") if record.get("pdf_page_index") is not None else (int(parsed.group(2)) if parsed else None)
    if not catalogue_id or page_index is None:
        return record
    canvas = record.get("canvas_number")
    page_number = int(canvas) if canvas and str(canvas).isdigit() else page_index + 1
    record = dict(record)
    if re.fullmatch(r"diglit_\d+", str(catalogue_id)):
        doi = f"https://doi.org/10.11588/diglit.{str(catalogue_id).split('_', 1)[1]}"
        record["source_catalogue_url"] = doi
        record["source_page_url"] = f"{doi}#{page_number:04d}"
        record["source_image_url"] = record["source_page_url"]
        record["pdf_url"] = doi
        return record
    record.setdefault("source_catalogue_url", f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}")
    record.setdefault("source_page_url", f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}/{page_number:04d}")
    record.setdefault("source_image_url", f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}/{page_number:04d}/max")
    record.setdefault("pdf_url", f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}/download.pdf")
    return record

def result_card(rank, score, stem, path, provenance):
    with Image.open(path) as image:
        buffer = io.BytesIO(); image.convert("RGB").save(buffer, format="JPEG", quality=88)
    source = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    links = [html.A("Image Source", href=provenance.get("source_image_url"), target="_blank", rel="noreferrer")]
    card = html.Div([html.Button([html.Img(src=source, alt=f"Rank {rank}: {stem}"), html.Div([html.Span(f"{rank:02d}", className="rank")], className="card-meta"), html.P(stem, className="record-id"), html.Span("Inspect", className="inspect-label")], id={"type": "result-card", "rank": rank}, n_clicks=0, className="result-card-open"), html.Div(links, className="result-links")], className="result-card")
    return card, {"src": source, "stem": stem, "score": f"{score:.3f}", "rank": rank, **provenance}

def decode_upload(contents):
    if not contents or "," not in contents:
        raise ValueError("Invalid upload payload")
    _, encoded = contents.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")

app = Dash(__name__, title="The Auction Dome")
@app.server.route("/archive-stills/<path:filename>")
def archive_still(filename):
    return send_from_directory(QUERY_IMAGE_DIR, filename)

@app.server.route("/catalogue-pages/<path:filename>")
def catalogue_page(filename):
    return send_from_directory(PDF_IMAGE_DIR, filename)

app.layout = html.Main([
    dcc.Store(id="result-data", data=[]),
    dcc.Store(id="inspector-state", data={"open": False, "rank": None, "rotation": 0, "zoom": 1}),
    dcc.Store(id="inspector-key", data={}),
    html.Header([
        html.Div([html.Span("A", className="monogram"), html.Span("The Auction Dome")], className="wordmark"),
        html.Div([html.Span("German Sales · 1901–1945"), html.Span("Visual index")], className="edition"),
    ], className="masthead"),
    html.Section([
        html.Div([
            html.P("Unsupervised image retrieval for auction catalogues", className="eyebrow"),
            html.H1(["Find the", html.Br(), "image before", html.Br(), "the record."]),
            html.P("A visual retrieval system for historical auction catalogues when titles, names, and OCR are incomplete.", className="lead"),
            html.A("Choose a test image ↓", href="#test-images", className="hero-link"),
        ], className="hero-copy"),
        html.Div([
            html.Img(src="/catalogue-pages/dorotheum1935_09_26_page83.jpg", className="catalogue-sheet", alt="Full auction catalogue page"),
            html.Div([html.Span("01  Degraded print"), html.Span("02  Frozen descriptor"), html.Span("03  Exact visual ranking")], className="thesis-stack"),
            html.Div([html.Span("Page 83"), html.Span("Dorotheum · 1935")], className="sheet-caption"),
            html.Div(className="orbit orbit-one"), html.Div(className="orbit orbit-two"),
        ], className="hero-image"),
        html.Div([html.Span("589,501"), html.Span("Catalogue crops · 16,469 PDFs")], className="hero-stat"),
        html.P("Scroll / select / compare", className="hero-side"),
    ], className="hero"),
    html.Section([
        html.P("Start with a visual fragment", className="eyebrow"),
        html.Div([html.H2("Twelve works. Twelve ways into the record."), html.P("These original query images are ready to use. Select one to load it into the research table below, or bring your own image.")], className="library-head"),
        html.Div([html.Button([html.Img(src=f"/archive-stills/{filename}", alt=label), html.Span(label), html.Small(filename.rsplit(".", 1)[0])], id={"type": "sample-query", "sample": filename}, className="sample-card", n_clicks=0) for filename, label in SAMPLE_QUERIES], className="sample-grid"),
    ], className="sample-library", id="test-images"),
    html.Section([
        dcc.Tabs(id="info-tabs", value="guide", children=[
            dcc.Tab(label="How to read this", value="guide", className="info-tab", selected_className="info-tab--selected"),
            dcc.Tab(label="Research note", value="research", className="info-tab", selected_className="info-tab--selected"),
        ], className="info-tabs"),
        html.Div(id="info-tab-content", className="info-tab-content"),
    ], className="method-band"),
    html.Section([
        html.Div([
            html.P("Research table", className="eyebrow"), html.H2("Ask the archive a visual question."),
            html.Div([html.Img(src="/catalogue-pages/boerner1921_11_08bd1_page38.jpg", alt="Full auction catalogue page"), html.Div([html.Span("Visual evidence"), html.Span("before metadata")])], className="table-image"),
        ], className="table-intro"),
        html.Section([
            html.Div([html.Span("Your selected query"), html.Span(id="sample-label", children="Select a test image or upload")], className="field-heading"),
            dcc.Loading(type="default", custom_spinner=html.Span("Loading image…", className="loading-message"), children=html.Div([html.Img(id="query-preview", alt="Selected query preview"), html.Span("No query selected", id="query-placeholder")], className="query-preview")),
            dcc.Upload(html.Div([html.Span("Bring your own image"), html.Small("JPG, PNG, or WEBP · drop or select a file")]), id="query-image", className="upload", accept="image/*", multiple=False),
            html.Div([html.Label("Retrieval method"), dcc.Dropdown([{"label": key, "value": key} for key in CONFIGS], "C5 Only", id="method", clearable=False)], className="control"),
            html.Div([html.Label("Visual neighbours"), dcc.Slider(1, 60, 1, value=12, id="top-k", marks={1:"1", 12:"12", 60:"60"})], className="control"),
            html.Button([html.Span("Compare visual neighbours"), html.Span("→")], id="search", n_clicks=0, className="search-button"),
            html.P("Loading visual matches…", id="loading-message", className="loading-message", style={"display": "none"}),
            html.P("A selected test image is immediately ready for comparison.", className="quiet-note"),
        ], className="query-panel"),
        html.Section([
            html.Div([html.P("Visual matches", className="eyebrow"), html.Span("Sorted by similarity", className="result-key")], className="result-heading"),
            dcc.Loading(type="circle", color="#916949", custom_spinner=html.Span("Loading visual matches…", className="loading-message"), children=html.Div([html.P("Select a study above.", className="empty-title"), html.P("Then the evidence board will arrange the closest visual neighbours here.")], id="results", className="results")),
            html.P("Ready — select a study or upload an image.", id="status", className="status"),
        ], className="results-panel"),
    ], className="research-table", id="research-desk"),
    html.Div([
        html.Button("Close", id="inspect-close", n_clicks=0, className="inspect-close"),
        html.Div([html.Img(id="inspect-image", alt="Expanded visual match")], className="inspect-stage"),
        html.Div([html.Div(id="inspect-caption", className="inspect-caption"), html.Div([
            html.Button("←", id="inspect-prev", n_clicks=0, className="inspect-control", title="Previous result"),
            html.Button("→", id="inspect-next", n_clicks=0, className="inspect-control", title="Next result"),
            html.Button("↶", id="inspect-rotate-left", n_clicks=0, className="inspect-control", title="Rotate left"),
            html.Button("−", id="inspect-zoom-out", n_clicks=0, className="inspect-control", title="Zoom out"),
            html.Button("+", id="inspect-zoom-in", n_clicks=0, className="inspect-control", title="Zoom in"),
            html.Button("↷", id="inspect-rotate-right", n_clicks=0, className="inspect-control", title="Rotate right"),
        ], className="inspect-controls")], className="inspect-footer"),
    ], id="inspector", className="inspector"),
    html.Footer([html.Span("The Auction Dome"), html.Span("Visual retrieval for auction-catalogue research"), html.Span("Evidence ≠ attribution")], className="footer")
], className="page")

@app.callback(Output("query-image", "contents"), Input({"type": "sample-query", "sample": ALL}, "n_clicks"), prevent_initial_call=True)
def load_sample_query(_):
    filename = ctx.triggered_id["sample"]
    with open(os.path.join(QUERY_IMAGE_DIR, filename), "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@app.callback(Output("query-preview", "src"), Output("query-preview", "style"), Output("query-placeholder", "style"), Output("sample-label", "children"), Input("query-image", "contents"), Input("query-image", "filename"))
def show_query_preview(contents, filename):
    if not contents:
        return None, {"display": "none"}, {}, "Select a test image or upload"
    try:
        decode_upload(contents)
    except (ValueError, OSError, base64.binascii.Error):
        return None, {"display": "none"}, {}, "Upload failed — choose a valid image"
    return contents, {"display": "block"}, {"display": "none"}, f"Image loaded — {filename or 'ready to compare'}"

@app.callback(Output("info-tab-content", "children"), Input("info-tabs", "value"))
def render_info_tab(value):
    if value == "research":
        return html.Div([
            html.P("Unsupervised image retrieval for auction catalogues", className="principle"),
            html.P("This thesis investigates whether visual similarity can help researchers locate related catalogue records when titles, names, and OCR are unreliable."),
            html.Div([
                html.Div([html.Span("Corpus"), html.P("German Sales 1901–1945 · 16,469 PDFs")]),
                html.Div([html.Span("Representation"), html.P("Self-supervised visual features · C5 and C4+C5 indexes")]),
                html.Div([html.Span("Search"), html.P("Exact nearest-neighbour ranking over 589,501 precomputed catalogue crops")]),
            ], className="research-facts"),
            html.P("Thesis finding: self-supervised representations provide a practical zero-shot route into large historical catalogues without annotated training data.", className="research-finding"),
        ], className="method-note research-note")
    return html.Div([
        html.Div([html.Span("01  Visual query"), html.P("Select one of the studies or upload a catalogue detail.")], className="guide-step"),
        html.Div([html.Span("02  Ranked neighbours"), html.P("Compare the closest visual pages returned by the indexed corpus.")], className="guide-step"),
        html.Div([html.Span("03  Research lead"), html.P("Inspect the evidence and follow the catalogue record for further research.")], className="guide-step"),
    ], className="method-note guide-note")

@app.callback(Output("inspector-state", "data"), Input({"type": "result-card", "rank": ALL}, "n_clicks"), Input("inspect-close", "n_clicks"), Input("inspect-prev", "n_clicks"), Input("inspect-next", "n_clicks"), Input("inspect-rotate-left", "n_clicks"), Input("inspect-rotate-right", "n_clicks"), Input("inspect-zoom-in", "n_clicks"), Input("inspect-zoom-out", "n_clicks"), Input("inspector-key", "data"), State("inspector-state", "data"), State("result-data", "data"), prevent_initial_call=True)
def update_inspector(_, __, prev, next_, ___, ____, _____, ______, key_event, state, records):
    state = state or {"open": False, "rank": None, "rotation": 0, "zoom": 1}
    trigger = ctx.triggered_id
    if isinstance(trigger, dict):
        click_value = next((item.get("value") for item in ctx.triggered if str(item.get("prop_id", "")).endswith(".n_clicks")), 0)
        clicked = any((value or 0) > 0 for value in click_value) if isinstance(click_value, list) else bool(click_value)
        if not clicked: return state
        return {"open": True, "rank": trigger["rank"], "rotation": 0, "zoom": 1}
    if trigger == "inspect-close": state["open"] = False
    if trigger == "inspector-key":
        key = (key_event or {}).get("key")
        if key == "Escape": state["open"] = False
        if key in ("ArrowLeft", "ArrowRight") and state.get("open") and records:
            ranks = [item["rank"] for item in records]
            offset = -1 if key == "ArrowLeft" else 1
            current = state.get("rank") if state.get("rank") in ranks else ranks[0]
            state["rank"] = ranks[max(0, min(len(ranks) - 1, ranks.index(current) + offset))]
            state["rotation"], state["zoom"] = 0, 1
    if trigger in ("inspect-prev", "inspect-next") and records:
        ranks = [item["rank"] for item in records]
        offset = -1 if trigger == "inspect-prev" else 1
        current = state.get("rank") if state.get("rank") in ranks else ranks[0]
        state["rank"] = ranks[max(0, min(len(ranks) - 1, ranks.index(current) + offset))]
        state["rotation"], state["zoom"] = 0, 1
    if trigger == "inspect-rotate-left": state["rotation"] = (state["rotation"] - 90) % 360
    if trigger == "inspect-rotate-right": state["rotation"] = (state["rotation"] + 90) % 360
    if trigger == "inspect-zoom-in": state["zoom"] = min(3, state["zoom"] + .25)
    if trigger == "inspect-zoom-out": state["zoom"] = max(1, state["zoom"] - .25)
    return state

@app.callback(Output("inspector", "className"), Output("inspect-image", "src"), Output("inspect-image", "style"), Output("inspect-caption", "children"), Input("inspector-state", "data"), State("result-data", "data"))
def render_inspector(state, records):
    if not state or not state["open"] or not records: return "inspector", None, {}, ""
    record = next((item for item in records if item["rank"] == state["rank"]), None)
    if not record: return "inspector", None, {}, ""
    style = {"transform": f"rotate({state['rotation']}deg) scale({state['zoom']})"}
    caption = [html.Span(f"Rank {record['rank']:02d}", className="inspect-rank"), html.Span(record["stem"], className="inspect-stem"), html.Div([html.A("Image Source", href=record.get("source_image_url"), target="_blank", rel="noreferrer")], className="inspect-links"), html.Span("←/→ navigate · Esc close · +/− zoom · ↶/↷ rotate", className="inspect-hint")]
    return "inspector is-open", record["src"], style, caption

@app.callback(Output("results", "children"), Output("status", "children"), Output("result-data", "data"), Input("search", "n_clicks"), State("query-image", "contents"), State("method", "value"), State("top-k", "value"), prevent_initial_call=True, running=[(Output("loading-message", "style"), {"display": "block"}, {"display": "none"}), (Output("search", "disabled"), True, False), (Output("query-image", "disabled"), True, False)])
def search(_, contents, method, top_k):
    if not contents: return html.Div([html.P("Choose an image first.", className="empty-title"), html.P("The archive is ready when your query is.")]), "No query image selected.", []
    try:
        query = decode_upload(contents)
        state = get_state()
        load_features(state, method)
        scores, indices = state.indexes[method].search(encode(state, query, state.configs_by_method[method], method), int(top_k)); cards, records = [], []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), 1):
            stem = state.stems_by_method[method][index]; path = state.paths.get(stem) or state.base_paths.get(DIR_RE.sub("", stem))
            if path and os.path.exists(path):
                card, record = result_card(rank, float(score), stem, path, provenance_for(state, stem)); cards.append(card); records.append(record)
        return html.Div(cards, className="result-grid"), f"{len(cards)} visual neighbours returned · {method} · {DEVICE}", records
    except (ValueError, OSError, UnidentifiedImageError, base64.binascii.Error):
        return html.Div([html.P("This image could not be read.", className="empty-title"), html.P("Use a JPG, PNG, or WEBP image, then try again.")]), "Upload failed. Choose a valid image file.", []
    except Exception:
        return html.Div([html.P("The archive could not open.", className="empty-title"), html.P("Check the configured data paths, then try again.")]), "Search unavailable. Check the configured data paths.", []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-image-dir", default=os.environ.get("DINO_DB_IMAGE_DIR", DEFAULT_IMAGE_DIR))
    parser.add_argument("--features-base", default=os.environ.get("DINO_FEATURES_BASE"))
    parser.add_argument("--host", default=os.environ.get("DINO_APP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DINO_APP_PORT", "7865")))
    args = parser.parse_args()
    if args.db_image_dir: os.environ["DINO_DB_IMAGE_DIR"] = args.db_image_dir
    if args.features_base: os.environ["DINO_FEATURES_BASE"] = args.features_base
    state = get_state()
    for method in CONFIGS: load_features(state, method)
    app.run(host=args.host, port=args.port, debug=False)
