import html
import io

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"
APP_NAME = "Khalil Plant Disease Detection"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Accepted upload types. HEIC/HEIF (iPhone photos) need `pillow-heif` in requirements.txt.
IMAGE_TYPES = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "tif", "tiff", "gif"]
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    IMAGE_TYPES += ["heic", "heif"]
except Exception:
    pass

# The crops the model was trained on (from the training dataset).
CROPS = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
    "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
]

# Leaf check: the disease model only knows leaves, so it would label ANY image (a person,
# a car...) as some disease. A small zero-shot CLIP model screens those out first.
USE_LEAF_CHECK = True      # set to False to skip the check (saves memory)
LEAF_MIN_SCORE = 0.40      # minimum "looks like a leaf or plant" probability
MIN_CONFIDENCE = 55.0      # below this, the disease prediction is shown as low confidence
CLIP_ID = "openai/clip-vit-base-patch32"

LEAF_PROMPTS = [
    "a close-up photo of a plant leaf",
    "a photo of a diseased plant leaf",
    "a photo of a green crop plant",
]
OTHER_PROMPTS = {
    "a photo of a person's face": "a person",
    "a photo of a person": "a person",
    "a selfie": "a person",
    "a photo of a hand": "a hand",
    "a photo of an animal": "an animal",
    "a photo of food": "food",
    "a photo of a building": "a building",
    "a photo of a car or vehicle": "a vehicle",
    "a photo of a room with furniture": "an indoor scene",
    "a screenshot or a document with text": "a document or screenshot",
    "a photo of the sky or a landscape": "a landscape",
}

# ------------------------------------------------------------------
# Themes: System follows the device, Light and Dark are forced.
# ------------------------------------------------------------------
LIGHT = """
  --bg:#F2F7EF;
  --blob1:rgba(184,232,132,.60); --blob2:rgba(110,212,180,.42); --blob3:rgba(255,230,160,.38);
  --glass:rgba(255,255,255,.70); --glass-solid:#FFFFFF; --soft:rgba(15,140,96,.07);
  --line:rgba(22,70,48,.13);
  --ink:#10241B; --muted:#587063;
  --accent:#0E8F62; --accent-ink:#0A6E4A; --accent-soft:rgba(14,143,98,.11); --dash:rgba(14,143,98,.50);
  --grad-a:#0B8A5C; --grad-b:#4CAE3A;
  --btn-a:#0B8A5C; --btn-b:#1F8F4E; --btn-ink:#FFFFFF;
  --track:rgba(16,36,27,.09); --fill-dim:rgba(14,143,98,.34);
  --ok-1:#D8F4E3; --ok-2:#EBF9CF; --ok-line:#A5DDBB; --ok-ring:#0E8F62;
  --bad-1:#FFE5D6; --bad-2:#FFF0D4; --bad-line:#F2BFA0; --bad-ring:#D1531C;
  --shadow:0 1px 0 rgba(255,255,255,.8) inset, 0 14px 44px rgba(16,70,45,.10);
"""

DARK = """
  --bg:#06110D;
  --blob1:rgba(22,120,80,.46); --blob2:rgba(10,110,122,.34); --blob3:rgba(112,150,36,.16);
  --glass:rgba(16,36,28,.64); --glass-solid:#10241C; --soft:rgba(61,220,151,.07);
  --line:rgba(255,255,255,.10);
  --ink:#E9F5EE; --muted:#95B0A2;
  --accent:#3DDC97; --accent-ink:#7CEBB6; --accent-soft:rgba(61,220,151,.13); --dash:rgba(61,220,151,.50);
  --grad-a:#3DDC97; --grad-b:#B9F26B;
  --btn-a:#3DDC97; --btn-b:#8FE36B; --btn-ink:#06251A;
  --track:rgba(255,255,255,.10); --fill-dim:rgba(61,220,151,.38);
  --ok-1:rgba(61,220,151,.17); --ok-2:rgba(185,242,107,.08); --ok-line:rgba(61,220,151,.38); --ok-ring:#3DDC97;
  --bad-1:rgba(255,128,76,.18); --bad-2:rgba(255,190,90,.08); --bad-line:rgba(255,128,76,.42); --bad-ring:#FF8F5C;
  --shadow:0 1px 0 rgba(255,255,255,.05) inset, 0 14px 44px rgba(0,0,0,.45);
"""

mode = st.session_state.get("theme", "System")
if mode == "Light":
    theme_vars = f":root {{ {LIGHT} color-scheme: light; }}"
elif mode == "Dark":
    theme_vars = f":root {{ {DARK} color-scheme: dark; }}"
else:
    theme_vars = (
        f":root {{ {LIGHT} color-scheme: light dark; }} "
        f"@media (prefers-color-scheme: dark) {{ :root {{ {DARK} }} }}"
    )

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
/*THEME_VARS*/
:root { --radius:20px; }
html, body, .stApp { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp {
 background:
   radial-gradient(ellipse at 88% 4%, rgba(105,153,94,.14), transparent 32%),
   linear-gradient(180deg,var(--bg),var(--bg));
 background-attachment:fixed;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none!important}
header[data-testid="stHeader"] {background:transparent;height:0}
.block-container {max-width:1280px;padding:1.1rem 2rem 3.5rem}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif;letter-spacing:-.035em}
[data-testid="stRadio"] {display:flex;justify-content:flex-end}
[data-testid="stRadio"]>label {display:none}
[data-testid="stRadio"] [role="radiogroup"] {
 display:inline-flex;gap:.2rem;padding:.28rem;border-radius:13px;
 background:var(--glass-solid);border:1px solid var(--line);box-shadow:0 4px 18px rgba(15,40,28,.06)
}
[data-testid="stRadio"] [role="radiogroup"] label {margin:0;padding:.35rem .75rem;border-radius:9px}
[data-testid="stRadio"] [role="radiogroup"] label p {font-size:.8rem;font-weight:700;color:var(--muted)}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {background:var(--accent-soft)}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {color:var(--accent-ink)}
.brandbar {display:flex;align-items:center;justify-content:space-between;padding:.25rem 0 1rem;border-bottom:1px solid var(--line)}
.brand {display:flex;align-items:center;gap:.7rem}
.brand .mark {width:40px;height:40px;border-radius:13px;display:grid;place-items:center;background:#123E2D;color:#DDF5D7;box-shadow:0 7px 18px rgba(18,62,45,.18)}
.brand .mark svg {width:24px;height:24px}
.brand b {font:800 1rem 'Manrope';letter-spacing:-.03em}
.brand small {display:block;color:var(--muted);font-size:.72rem;margin-top:1px}
.top-label {font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.hero {display:grid;grid-template-columns:1.35fr .65fr;gap:2rem;align-items:stretch;padding:2.1rem 0 1.5rem}
.hero-main {padding:1.5rem 0}
.eyebrow {display:inline-flex;align-items:center;gap:.5rem;font-size:.76rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:var(--accent-ink)}
.eyebrow:before {content:'';width:23px;height:2px;background:var(--accent);border-radius:4px}
.hero h1 {font-size:clamp(2rem,4vw,3.35rem);line-height:1.08;font-weight:800;margin:.85rem 0 .8rem;max-width:15ch;color:var(--ink)}
.hero h1 em {font-style:normal;color:var(--accent)}
.hero p {max-width:58ch;font-size:1rem;line-height:1.7;color:var(--muted);margin:0}
.hero-panel {position:relative;overflow:hidden;border-radius:24px;padding:1.4rem 1.5rem;background:#143D2E;color:#F2F7EE;min-height:190px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 14px 34px rgba(13,55,38,.15)}
.hero-panel:after {content:'✳';position:absolute;right:12px;bottom:-55px;font-size:180px;line-height:1;color:rgba(220,245,211,.07);transform:rotate(-15deg)}
.panel-top {display:flex;justify-content:space-between;align-items:center;position:relative;z-index:1}
.panel-kicker {font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:#B8D9C4;font-weight:800}
.panel-dot {width:9px;height:9px;border-radius:50%;background:#A9E27D;box-shadow:0 0 0 5px rgba(169,226,125,.13)}
.panel-stat {position:relative;z-index:1;display:flex;align-items:end;gap:.7rem}
.panel-stat strong {font:800 2.3rem 'Manrope';letter-spacing:-.06em}
.panel-stat span {font-size:.83rem;color:#C2D8C9;padding-bottom:.45rem}
.panel-foot {position:relative;z-index:1;font-size:.78rem;color:#B8D9C4}
.crop-strip {display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;padding:.85rem 1rem;margin:.15rem 0 1.6rem;background:var(--glass);border:1px solid var(--line);border-radius:15px}
.crop-strip .lbl {font-size:.75rem;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-right:.3rem}
.crop {font-size:.76rem;font-weight:600;color:var(--ink);padding:.28rem .65rem;background:var(--soft);border:1px solid var(--line);border-radius:8px}
.card {background:var(--glass-solid);border:1px solid var(--line);border-radius:22px;padding:1.4rem;box-shadow:0 8px 28px rgba(19,49,34,.055);color:var(--ink);animation:rise .4s ease both}
@keyframes rise {from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.head {display:flex;align-items:center;gap:.75rem;margin-bottom:1.2rem}
.head .ico {width:39px;height:39px;border-radius:12px;display:grid;place-items:center;background:var(--accent-soft);font-size:1.1rem}
.head h3 {margin:0;font-size:1rem;font-weight:800;color:var(--ink)}
.head .sub {margin:.15rem 0 0;color:var(--muted);font-size:.79rem;line-height:1.45}
[data-testid="stColumn"]:has(.upload-anchor),[data-testid="column"]:has(.upload-anchor) {background:var(--glass-solid);border:1px solid var(--line);border-radius:22px;padding:1.4rem!important;box-shadow:0 8px 28px rgba(19,49,34,.055);box-sizing:border-box}
[data-testid="stFileUploader],[data-testid="stFileUploader"] * {color:var(--ink)!important}
[data-testid="stFileUploader"] small,[data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stFileUploaderDropzoneInstructions"] * {color:var(--muted)!important}
[data-testid="stFileUploaderDropzone"],[data-testid="stFileUploader"] section {background:var(--soft)!important;border:1.5px dashed var(--dash)!important;border-radius:16px!important;padding:1.35rem 1rem!important}
[data-testid="stFileUploaderDropzone"]:hover,[data-testid="stFileUploader"] section:hover {border-color:var(--accent)!important;background:var(--accent-soft)!important}
[data-testid="stFileUploaderDropzone"] button {background:#174B36!important;border:0!important;border-radius:9px!important;font-weight:700;padding:.42rem 1rem!important}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploaderDropzone"] button * {color:#fff!important}
[data-testid="stFileUploaderFile"] {background:var(--soft)!important;border:1px solid var(--line);border-radius:11px;padding:.4rem .65rem;margin-top:.6rem}
[data-testid="stFileUploaderFile"] *,[data-testid="stFileUploaderFileName"] {color:var(--ink)!important;opacity:1!important}
[data-testid="stImage"] img {border-radius:15px;border:1px solid var(--line);box-shadow:0 5px 20px rgba(10,35,20,.08)}
[data-testid="stImageCaption"],[data-testid="stCaptionContainer"] {color:var(--muted)!important}
[data-testid="stSpinner"],[data-testid="stSpinner"] * {color:var(--ink)!important}
[data-testid="stAlert"] {background:var(--soft)!important;border:1px solid var(--line);border-radius:13px}
[data-testid="stAlert"],[data-testid="stAlert"] * {color:var(--ink)!important}
[data-testid="stExpander"] details {background:var(--soft);border:1px solid var(--line);border-radius:13px}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {color:var(--ink)!important}
pre.err {white-space:pre-wrap;word-break:break-word;font-size:.82rem;line-height:1.5;background:var(--soft);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:.8rem}
.verdict {border-radius:17px;padding:1.1rem;display:flex;gap:1.1rem;align-items:center;margin-bottom:1rem}
.verdict.healthy {background:linear-gradient(120deg,var(--ok-1),var(--ok-2));border:1px solid var(--ok-line);--ring:var(--ok-ring)}
.verdict.diseased {background:linear-gradient(120deg,var(--bad-1),var(--bad-2));border:1px solid var(--bad-line);--ring:var(--bad-ring)}
.verdict.unsure {background:var(--soft);border:1px solid var(--line);--ring:var(--fill-dim)}
.ring {--p:0;flex:none;width:94px;height:94px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--ring) calc(var(--p)*1%),var(--track) 0)}
.ring>div {width:76px;height:76px;border-radius:50%;background:var(--glass-solid);color:var(--ink);display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.05}
.ring b {font:800 1.25rem 'Manrope'} .ring small {font-size:.6rem;color:var(--muted);margin-top:.15rem}
.noicon {flex:none;width:94px;height:94px;border-radius:50%;display:grid;place-items:center;font-size:2.3rem;background:var(--glass-solid);border:1px solid var(--line)}
.verdict .plant {margin:0;color:var(--muted);font-size:.8rem;font-weight:600}
.verdict .name {font:800 1.35rem 'Manrope';line-height:1.2;letter-spacing:-.03em;margin:.2rem 0 .55rem;color:var(--ink)}
.badge {display:inline-block;padding:.25rem .65rem;border-radius:7px;font-size:.72rem;font-weight:800;color:#fff;background:#566B5F}
.healthy .badge {background:#24764A}.diseased .badge {background:#B8461B}
.note {display:flex;gap:.65rem;align-items:flex-start;padding:.75rem .85rem;border-radius:12px;background:var(--soft);border:1px solid var(--line);color:var(--ink);font-size:.85rem;line-height:1.55;margin-bottom:1.1rem}
.note .e {font-size:1rem}
.list-title {font-weight:800;font-size:.88rem;margin:0 0 .2rem;color:var(--ink)}
.bar-row {margin:.8rem 0}
.bar-head {display:flex;justify-content:space-between;gap:1rem;font-size:.82rem;margin-bottom:.35rem;color:var(--ink)}
.bar-head b {font-weight:700}.bar-head span {color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.track {height:7px;border-radius:99px;background:var(--track);overflow:hidden}
.fill {height:100%;border-radius:99px;background:var(--fill-dim);animation:grow .7s ease both}
.fill.top {background:linear-gradient(90deg,var(--grad-a),var(--grad-b))}
@keyframes grow {from{width:0}}
.disclaimer {margin-top:1rem;font-size:.76rem;color:var(--muted);line-height:1.5}
.steps {display:grid;gap:.6rem}
.step {display:flex;gap:.8rem;align-items:center;padding:.8rem;background:var(--soft);border:1px solid var(--line);border-radius:13px}
.step .n {flex:none;width:29px;height:29px;border-radius:9px;display:grid;place-items:center;background:#174B36;color:#fff;font-weight:800;font-size:.8rem}
.step b {display:block;color:var(--ink);font-size:.86rem;font-weight:800}
.step span {color:var(--muted);font-size:.79rem;line-height:1.4}
.section-label {display:flex;align-items:center;gap:.6rem;margin:1.8rem 0 .8rem;font-size:.74rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:var(--muted)}
.section-label:after {content:'';height:1px;flex:1;background:var(--line)}
.tips {display:grid;grid-template-columns:repeat(3,1fr);gap:.85rem;margin-top:.3rem}
.tip {background:var(--glass-solid);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;box-shadow:0 5px 18px rgba(19,49,34,.04)}
.tip .e {width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:var(--accent-soft);font-size:1rem}
.tip h4 {margin:.65rem 0 .2rem;font-size:.9rem;font-weight:800;color:var(--ink)}
.tip p {margin:0;font-size:.8rem;color:var(--muted);line-height:1.5}
.footer {text-align:center;color:var(--muted);font-size:.78rem;margin-top:2rem;line-height:1.8}
.footer a {color:var(--accent-ink);font-weight:800;text-decoration:none}
@media(max-width:850px){
 .block-container{padding:1rem 1.15rem 2.5rem}
 .hero{grid-template-columns:1fr;gap:1rem;padding:1.35rem 0 1rem}
 .hero-main{padding:.6rem 0}
 .hero-panel{min-height:138px}
 .tips{grid-template-columns:1fr}
}
@media(max-width:700px){
 [data-testid="stHorizontalBlock"]{flex-direction:column!important;align-items:stretch!important;gap:1rem!important}
 [data-testid="column"],[data-testid="stColumn"]{width:100%!important;min-width:100%!important;flex:1 1 100%!important}
 [data-testid="stRadio"]{justify-content:flex-start!important}
 [data-testid="stRadio"] [role="radiogroup"]{width:100%;justify-content:space-between}
 [data-testid="stRadio"] [role="radiogroup"] label{flex:1;text-align:center;padding:.45rem .25rem}
 .block-container{padding:.8rem .85rem 2rem}
 .hero h1{font-size:clamp(1.8rem,8vw,2.5rem);max-width:18ch}
 .hero p{font-size:.92rem}
 .crop-strip{gap:.35rem;padding:.75rem}
 .crop{font-size:.72rem}
 .card{padding:1rem;border-radius:17px}
 [data-testid="stColumn"]:has(.upload-anchor),[data-testid="column"]:has(.upload-anchor){padding:1rem!important}
 .verdict{gap:.8rem}
 [data-testid="stImage"] img{width:100%!important;height:auto!important}
 [data-testid="stFileUploaderDropzone"],[data-testid="stFileUploader"] section{padding:1rem .6rem!important}
}
@media(max-width:420px){
 .brand b{font-size:.9rem}
 .brand .mark{width:36px;height:36px}
 .verdict{flex-direction:column;text-align:center}
 .ring,.noicon{width:84px;height:84px}
 .ring>div{width:68px;height:68px}
 .hero-panel{padding:1.1rem;min-height:130px}
 .panel-stat strong{font-size:1.9rem}
}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
:focus-visible{outline:3px solid var(--accent)!important;outline-offset:2px}\n[data-testid="stAppViewContainer"]{overflow-x:hidden}\n[data-testid="stFileUploaderFileName"]{overflow-wrap:anywhere;word-break:break-word}
</style>
""".replace("/*THEME_VARS*/", theme_vars)

st.markdown(CSS, unsafe_allow_html=True)

LEAF_SVG = """<svg viewBox="0 0 200 200" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M170 30C110 28 52 52 40 112c-4 22 4 44 20 56 22-6 44-18 62-40 24-30 38-62 48-98z"/><path d="M44 166C70 120 104 84 150 52"/><path d="M82 128l-2-30M100 106l-4-32M118 86l-2-28M72 138l-26-8M92 118l-24-14M112 98l-22-12M132 78l-18-8"/></svg>"""


def compact(markup: str) -> str:
    """Collapse multi-line HTML to one line so Markdown never mistakes indentation for code."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


# ------------------------------------------------------------------
# Models and helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor

        proc = CLIPProcessor.from_pretrained(CLIP_ID)
        clip = CLIPModel.from_pretrained(CLIP_ID)
        clip.eval()
        return proc, clip
    except Exception:
        return None


def load_image(image_bytes: bytes) -> Image.Image:
    """Open any supported image, fix phone rotation, flatten transparency onto white."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


@st.cache_data(show_spinner=False)
def check_leaf(image_bytes: bytes):
    """Return (is_leaf, what_it_looks_like, leaf_score)."""
    if not USE_LEAF_CHECK:
        return True, None, 1.0
    loaded = load_clip()
    if loaded is None:  # CLIP unavailable: fall back to the confidence check only
        return True, None, 1.0
    proc, clip = loaded
    prompts = LEAF_PROMPTS + list(OTHER_PROMPTS)
    inputs = proc(text=prompts, images=load_image(image_bytes), return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip(**inputs).logits_per_image.softmax(dim=1)[0]
    leaf_score = float(probs[: len(LEAF_PROMPTS)].sum())
    best_other = int(probs[len(LEAF_PROMPTS):].argmax())
    seen_as = list(OTHER_PROMPTS.values())[best_other]
    return leaf_score >= LEAF_MIN_SCORE, seen_as, leaf_score


@st.cache_data(show_spinner=False)
def predict(image_bytes: bytes, top_k: int = 5):
    processor, model = load_model()
    inputs = processor(images=load_image(image_bytes), return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    k = min(top_k, probs.shape[0])
    values, indices = torch.topk(probs, k)
    label_map = model.config.id2label

    results = []
    for p, i in zip(values.tolist(), indices.tolist()):
        label = label_map.get(i, label_map.get(str(i), str(i)))
        results.append((label, p * 100))
    return results


def split_label(raw: str):
    """Turn 'Tomato___Early_blight' into ('Tomato', 'Early blight')."""
    raw = str(raw)
    if "___" in raw:
        plant, cond = raw.split("___", 1)
    elif "__" in raw:
        plant, cond = raw.split("__", 1)
    elif "," in raw:
        plant, cond = raw.split(",", 1)
    else:
        plant, cond = "", raw
    clean = lambda s: " ".join(s.replace("_", " ").split()).strip(" ,")
    return clean(plant), clean(cond)


def is_healthy(cond: str) -> bool:
    return "healthy" in cond.lower()


def show_image(img, caption):
    """st.image across Streamlit versions (the width argument changed over time)."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except Exception:
            continue


def card_head(icon: str, title: str, sub: str) -> str:
    return (
        f'<div class="head"><div class="ico">{icon}</div>'
        f'<div><h3>{html.escape(title)}</h3><div class="sub">{html.escape(sub)}</div></div></div>'
    )


def empty_card_html() -> str:
    return compact(
        f"""
        <div class="card">
          {card_head("🔬", "Result", "Your diagnosis will appear here.")}
          <div class="steps">
            <div class="step"><div class="n">1</div><div><b>Upload a photo</b>
              <span>Choose a clear picture of one leaf.</span></div></div>
            <div class="step"><div class="n">2</div><div><b>The model analyses it</b>
              <span>It runs automatically as soon as the photo loads.</span></div></div>
            <div class="step"><div class="n">3</div><div><b>Read the result</b>
              <span>See the likely disease, the confidence and the other possibilities.</span></div></div>
          </div>
        </div>
        """
    )


def not_leaf_card_html(seen_as) -> str:
    looks = f"It looks more like {html.escape(seen_as)}. " if seen_as else ""
    return compact(
        f"""
        <div class="card">
          {card_head("🔬", "Result", "No diagnosis was made for this image")}
          <div class="verdict unsure">
            <div class="noicon">🚫</div>
            <div>
              <p class="plant">Image check</p>
              <div class="name">This doesn't look like a plant leaf</div>
              <span class="badge">Not a leaf</span>
            </div>
          </div>
          <div class="note"><span class="e">💡</span><span>{looks}Upload a close, clear photo of a single leaf
          from one of the supported crops and the model will check it for disease.</span></div>
        </div>
        """
    )


def result_card_html(results) -> str:
    top_label, top_conf = results[0]
    plant, condition = split_label(top_label)
    unsure = top_conf < MIN_CONFIDENCE

    if unsure:
        tone, badge = "unsure", "Low confidence"
        plant_line, name = "Not a confident match", "No clear match"
        list_title = "Closest matches (low confidence)"
        note_icon = "🔍"
        note = "The photo may be unclear, or it may show something the model doesn't know. Try a closer, sharper photo in daylight."
    elif is_healthy(condition):
        tone, badge = "healthy", "Looks healthy"
        plant_line, name = (f"{plant} leaf" if plant else "Predicted class"), condition[:1].upper() + condition[1:]
        list_title = f"Top {len(results)} predictions"
        note_icon = "🌱"
        note = "No signs of disease were found in this photo. Keep an eye on the plant and check again if the leaves change."
    else:
        tone, badge = "diseased", "Disease detected"
        plant_line, name = (f"{plant} leaf" if plant else "Predicted class"), condition[:1].upper() + condition[1:]
        list_title = f"Top {len(results)} predictions"
        note_icon = "⚠️"
        note = "Consider separating the plant and removing badly affected leaves, then confirm with a local agriculture expert before treating."

    bars = ""
    for rank, (label, conf) in enumerate(results):
        p_name, c_name = split_label(label)
        shown = f"{p_name} · {c_name}" if p_name else c_name
        cls = "fill top" if rank == 0 and not unsure else "fill"
        bars += (
            f'<div class="bar-row"><div class="bar-head"><b>{html.escape(shown)}</b><span>{conf:.2f}%</span></div>'
            f'<div class="track"><div class="{cls}" style="width:{max(conf, 0.8):.2f}%"></div></div></div>'
        )

    return compact(
        f"""
        <div class="card">
          {card_head("🔬", "Result", "Most likely class for this image")}
          <div class="verdict {tone}">
            <div class="ring" style="--p:{top_conf:.1f};"><div><b>{top_conf:.0f}%</b><small>confidence</small></div></div>
            <div>
              <p class="plant">{html.escape(plant_line)}</p>
              <div class="name">{html.escape(name)}</div>
              <span class="badge">{badge}</span>
            </div>
          </div>
          <div class="note"><span class="e">{note_icon}</span><span>{note}</span></div>
          <div class="list-title">{list_title}</div>
          {bars}
          <div class="disclaimer">This is a model prediction, not a confirmed diagnosis.</div>
        </div>
        """
    )


# ------------------------------------------------------------------
# Navigation and compact editorial hero
# ------------------------------------------------------------------
try:
    nav_left, nav_right = st.columns([3, 2], vertical_alignment="center")
except TypeError:
    nav_left, nav_right = st.columns([3, 2])

with nav_left:
    st.markdown(
        compact(
            f'<div class="brand"><div class="mark">{LEAF_SVG}</div>'
            f'<div><b>KHALIL’S <span style="color:var(--accent)">PLANT LAB</span></b>'
            f'<small>AI-powered crop health</small></div></div>'
        ),
        unsafe_allow_html=True,
    )
with nav_right:
    st.radio(
        "Theme",
        ["System", "Light", "Dark"],
        key="theme",
        horizontal=True,
        format_func=lambda x: {"System": "◐ System", "Light": "☼ Light", "Dark": "☾ Dark"}[x],
        label_visibility="collapsed",
    )

st.markdown(
    compact(
        """
        <div class="hero">
          <div class="hero-main">
            <div class="eyebrow">Plant health workspace</div>
            <h1>Khalil’s <em>Plant Disease</em> Detection</h1>
            <p>A smarter way to check your crops. Upload a leaf photo to identify likely diseases, review confidence and compare the closest predictions.</p>
          </div>
          <div class="hero-panel">
            <div class="panel-top"><span class="panel-kicker">AI diagnostic assistant</span><span class="panel-dot"></span></div>
            <div class="panel-stat"><strong>14</strong><span>supported crop types</span></div>
            <div class="panel-foot">Image screening · Disease classification · Confidence scoring</div>
          </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

crop_tags = "".join(f'<span class="crop">{c}</span>' for c in CROPS)
st.markdown(
    compact(f'<div class="crop-strip"><span class="lbl">Supported crops</span>{crop_tags}</div>'),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.markdown('<div class="section-label">New analysis <span>01 / 01</span></div>', unsafe_allow_html=True)
left, right = st.columns([1, 1], gap="large")

image = None
image_bytes = None
accepted = ", ".join(t.upper() for t in IMAGE_TYPES if t not in ("jpeg", "jfif", "tif"))

with left:
    st.markdown(
        '<div class="upload-anchor"></div>'
        + card_head("📷", "Upload a leaf photo", f"{accepted}. One leaf, in focus, on a plain background works best."),
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Leaf image", type=IMAGE_TYPES, label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = load_image(image_bytes)
            show_image(image, uploaded_file.name)
        except Exception:
            image = None
            image_bytes = None
            st.error("That file could not be read as an image. Try a JPG, PNG or WebP image instead.")

with right:
    if image is None:
        st.markdown(empty_card_html(), unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Analysing the image…"):
                is_leaf, seen_as, _leaf_score = check_leaf(image_bytes)
                results = predict(image_bytes) if is_leaf else None

            if not is_leaf:
                st.markdown(not_leaf_card_html(seen_as), unsafe_allow_html=True)
            else:
                st.markdown(result_card_html(results), unsafe_allow_html=True)
        except Exception as error:
            st.error("The model could not analyse this image. Check your connection and try another photo.")
            with st.expander("Technical details"):
                st.markdown(f'<pre class="err">{html.escape(repr(error))}</pre>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tips and footer
# ------------------------------------------------------------------
st.markdown('<div class="section-label">Get the best result</div>', unsafe_allow_html=True)
st.markdown(
    compact(
        f"""
        <div class="tips">
          <div class="tip"><div class="e">☀️</div><h4>Use natural light</h4>
            <p>Daylight without harsh shadows shows the true colour of spots and lesions.</p></div>
          <div class="tip"><div class="e">🍃</div><h4>Frame one leaf</h4>
            <p>Fill most of the frame with a single leaf. Avoid hands, soil and clutter.</p></div>
          <div class="tip"><div class="e">🎯</div><h4>Keep it sharp</h4>
            <p>Hold steady and tap to focus. Blurry photos lower the model's confidence.</p></div>
        </div>
        <div class="footer">Built with PyTorch, Hugging Face Transformers and Streamlit<br>
          <a href="https://huggingface.co/{MODEL_ID}" target="_blank" rel="noopener">View the model on Hugging Face</a></div>
        """
    ),
    unsafe_allow_html=True,
)
