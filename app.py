
import html
import io

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

DISEASE_MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"
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

# Optional HEIC/HEIF support (iPhone photos). Add `pillow-heif` to requirements.txt.
IMAGE_TYPES = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "tif", "tiff", "gif"]
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    IMAGE_TYPES += ["heic", "heif"]
except Exception:
    pass

# ------------------------------------------------------------------
# Themes (System follows the device; Light and Dark are forced)
# ------------------------------------------------------------------
LIGHT = """
  --bg:#EDF3EE; --surface:#FFFFFF; --surface2:#F2F8F3;
  --ink:#14261F; --muted:#556B5F;
  --line:#D4E2D8; --line-strong:#9FC2AB;
  --accent:#2F7D4F; --accent-2:#6BBE6F; --accent-soft:#DDF0E2;
  --track:#E1ECE4; --fill-dim:#A3C4B0;
  --ok-bg:#E4F5D3; --ok-line:#BFE29A; --ok-ring:#3F8F5B;
  --bad-bg:#FBEBE2; --bad-line:#EFC5AF; --bad-ring:#C0562B;
  --hero-a:#123B2E; --hero-b:#1B5240;
  --shadow:0 1px 2px rgba(20,38,31,.05), 0 8px 24px rgba(20,38,31,.06);
"""

DARK = """
  --bg:#0B1612; --surface:#12231C; --surface2:#182D24;
  --ink:#E7F3EA; --muted:#9BB3A5;
  --line:#25392F; --line-strong:#3A5546;
  --accent:#63C184; --accent-2:#B6E36B; --accent-soft:#1F3D2F;
  --track:#233A2F; --fill-dim:#4D7360;
  --ok-bg:#183323; --ok-line:#2D5A3A; --ok-ring:#63C184;
  --bad-bg:#37201A; --bad-line:#6A3A29; --bad-ring:#E98A5B;
  --hero-a:#0E2F24; --hero-b:#17503B;
  --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.35);
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
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

/*THEME_VARS*/

html, body, .stApp {
  font-family: 'DM Sans', system-ui, sans-serif;
  background: var(--bg);
  color: var(--ink);
}
.stApp { transition: background .2s ease; }

/* Hide Streamlit chrome so the theme always looks consistent */
#MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif; letter-spacing: -0.02em; }

/* ---------- Top bar ---------- */
.brand { display: flex; align-items: center; gap: .7rem; padding: .35rem 0; }
.brand .mark {
  width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--hero-a), var(--hero-b)); color: #CDEFA0;
}
.brand .mark svg { width: 24px; height: 24px; }
.brand b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.12rem; font-weight: 700; color: var(--ink); }

/* Theme switch (radio styled as a pill group) */
[data-testid="stRadio"] { display: flex; justify-content: flex-end; }
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] [role="radiogroup"] {
  display: inline-flex; gap: .15rem; padding: .25rem; border-radius: 999px;
  background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow);
}
[data-testid="stRadio"] [role="radiogroup"] label {
  margin: 0; padding: .32rem .8rem; border-radius: 999px; cursor: pointer;
}
[data-testid="stRadio"] [role="radiogroup"] label p {
  color: var(--ink); font-size: .88rem; font-weight: 600; white-space: nowrap;
}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background: var(--accent-soft); }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color: var(--accent); }

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 28px;
  padding: 3.1rem 3rem; margin: .6rem 0 1.6rem;
  background:
    radial-gradient(800px 360px at 92% -10%, rgba(182,227,107,.26), transparent 60%),
    linear-gradient(135deg, var(--hero-a) 0%, var(--hero-b) 100%);
  color: #F2FAEF;
}
.hero .art {
  position: absolute; right: -20px; bottom: -40px; width: 330px; height: 330px;
  color: #CDEFA0; opacity: .16; transform: rotate(-8deg); pointer-events: none;
}
.hero h1 {
  font-size: clamp(2.1rem, 4.6vw, 3.6rem); font-weight: 800; line-height: 1.03;
  margin: 0 0 .9rem; color: #fff; max-width: 15ch;
}
.hero p { font-size: 1.07rem; line-height: 1.55; max-width: 50ch; color: #CFE3D2; margin: 0; }

/* ---------- Cards ---------- */
.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 22px;
  padding: 1.5rem 1.6rem; box-shadow: var(--shadow); color: var(--ink);
}
.card h3 { font-size: 1.25rem; font-weight: 700; margin: 0 0 .2rem; color: var(--ink); }
.card .sub { color: var(--muted); font-size: .95rem; margin-bottom: 1rem; line-height: 1.45; }

/* The left column becomes the upload card */
[data-testid="stColumn"]:has(.upload-anchor),
[data-testid="column"]:has(.upload-anchor) {
  background: var(--surface); border: 1px solid var(--line); border-radius: 22px;
  padding: 1.5rem 1.6rem !important; box-shadow: var(--shadow); box-sizing: border-box;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"], [data-testid="stFileUploader"] * { color: var(--ink) !important; }
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] * {
  color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
  background: var(--surface2) !important; border: 2px dashed var(--line-strong) !important;
  border-radius: 18px !important; padding: 1.4rem !important;
  transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
}
/* Browse button */
[data-testid="stFileUploaderDropzone"] button {
  background: var(--accent) !important; border: 1px solid var(--accent) !important;
  border-radius: 999px !important; font-weight: 600;
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #fff !important; }
/* Uploaded file row: name, size, icon, remove button */
[data-testid="stFileUploaderFile"] {
  background: var(--surface) !important; border: 1px solid var(--line); border-radius: 12px;
  padding: .45rem .7rem; margin-top: .6rem;
}
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] {
  color: var(--ink) !important; opacity: 1 !important;
}
[data-testid="stFileUploaderFile"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderFile"] svg { fill: currentColor; }
[data-testid="stFileUploaderDeleteBtn"] button, [data-testid="stFileUploaderFile"] button {
  background: transparent !important; border: none !important;
}

/* ---------- Image / captions / spinner ---------- */
[data-testid="stImage"] img { border-radius: 18px; border: 1px solid var(--line); }
[data-testid="stImageCaption"], [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--ink) !important; }

/* ---------- Alerts and expanders ---------- */
[data-testid="stAlert"] {
  background: var(--surface2) !important; border: 1px solid var(--line-strong); border-radius: 14px;
}
[data-testid="stAlert"], [data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stExpander"] details {
  background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
pre.err {
  white-space: pre-wrap; word-break: break-word; font-size: .85rem; line-height: 1.5;
  background: var(--surface2); color: var(--ink); border: 1px solid var(--line);
  border-radius: 12px; padding: .8rem 1rem; margin: 0;
}

/* ---------- Result ---------- */
.verdict {
  border-radius: 20px; padding: 1.25rem 1.4rem; display: flex; gap: 1.2rem; align-items: center;
  margin-bottom: 1.3rem;
}
.verdict.healthy { background: var(--ok-bg); border: 1px solid var(--ok-line); --ring-color: var(--ok-ring); }
.verdict.diseased { background: var(--bad-bg); border: 1px solid var(--bad-line); --ring-color: var(--bad-ring); }
.ring {
  --p: 0; flex: none; width: 96px; height: 96px; border-radius: 50%;
  background: conic-gradient(var(--ring-color) calc(var(--p) * 1%), var(--track) 0);
  display: grid; place-items: center;
}
.ring span {
  width: 76px; height: 76px; border-radius: 50%; background: var(--surface); color: var(--ink);
  display: grid; place-items: center;
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.2rem;
}
.verdict .plant { color: var(--muted); font-size: .92rem; margin: 0; }
.verdict .name {
  font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.65rem; font-weight: 800;
  line-height: 1.1; margin: .12rem 0 .5rem; color: var(--ink);
}
.badge { display: inline-block; padding: .24rem .75rem; border-radius: 999px; font-size: .82rem; font-weight: 600; color: #fff; }
.healthy .badge { background: #2E7D4A; }
.diseased .badge { background: #B24A22; }
.verdict.unsure { background: var(--surface2); border: 1px solid var(--line-strong); --ring-color: var(--fill-dim); }
.unsure .badge { background: #5B6E63; }
.noicon {
  flex: none; width: 96px; height: 96px; border-radius: 50%; display: grid; place-items: center;
  font-size: 2.4rem; background: var(--surface); border: 1px solid var(--line);
}

.list-title { font-weight: 700; font-size: .98rem; margin: 0 0 .2rem; color: var(--ink); }
.bar-row { margin: .8rem 0; }
.bar-head { display: flex; justify-content: space-between; gap: 1rem; font-size: .92rem; margin-bottom: .32rem; color: var(--ink); }
.bar-head b { font-weight: 600; }
.bar-head span { color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.track { height: 9px; border-radius: 99px; background: var(--track); overflow: hidden; }
.fill { height: 100%; border-radius: 99px; background: var(--fill-dim); }
.fill.top { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.disclaimer { margin-top: 1.1rem; font-size: .85rem; color: var(--muted); line-height: 1.5; }

/* ---------- Empty state: how it works ---------- */
.steps { display: grid; gap: .8rem; margin-top: .4rem; }
.step {
  display: flex; gap: .9rem; align-items: flex-start; padding: .9rem 1rem;
  background: var(--surface2); border: 1px solid var(--line); border-radius: 16px;
}
.step .n {
  flex: none; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent); font-weight: 700; font-size: .9rem;
}
.step b { display: block; color: var(--ink); font-size: .98rem; }
.step span { color: var(--muted); font-size: .9rem; line-height: 1.45; }

/* ---------- Tips ---------- */
.tips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.6rem; }
.tip { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem; box-shadow: var(--shadow); }
.tip .emoji { font-size: 1.5rem; }
.tip h4 { font-size: 1.02rem; margin: .4rem 0 .2rem; color: var(--ink); }
.tip p { font-size: .9rem; color: var(--muted); margin: 0; line-height: 1.45; }

.footer { text-align: center; color: var(--muted); font-size: .85rem; margin-top: 2.2rem; line-height: 1.7; }
.footer a { color: var(--accent); font-weight: 600; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

@media (max-width: 800px) {
  .tips { grid-template-columns: 1fr; }
  .hero { padding: 2rem 1.4rem; }
  .hero .art { width: 220px; height: 220px; }
  .verdict { flex-direction: column; text-align: center; }
  [data-testid="stRadio"] { justify-content: flex-start; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
:focus-visible { outline: 3px solid var(--accent-2) !important; outline-offset: 2px; }
</style>
""".replace("/*THEME_VARS*/", theme_vars)

st.markdown(CSS, unsafe_allow_html=True)

LEAF_SVG = """
<svg viewBox="0 0 200 200" fill="none" stroke="currentColor" stroke-width="3"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M170 30C110 28 52 52 40 112c-4 22 4 44 20 56 22-6 44-18 62-40 24-30 38-62 48-98z"/>
  <path d="M44 166C70 120 104 84 150 52"/>
  <path d="M82 128l-2-30M100 106l-4-32M118 86l-2-28M72 138l-26-8M92 118l-24-14M112 98l-22-12M132 78l-18-8"/>
</svg>
"""


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


@st.cache_data(show_spinner=False)
def predict(image_bytes: bytes, top_k: int = 5):
    processor, model = load_model()
    image = load_image(image_bytes)

    inputs = processor(images=image, return_tensors="pt")
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


def load_image(image_bytes: bytes) -> Image.Image:
    """Open any supported image, fix phone rotation, flatten transparency onto white."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


# Leaf check: the disease model only knows plant leaves, so it will label ANY image
# (a person, a car...) as some disease. A small zero-shot CLIP model screens those out first.
USE_LEAF_CHECK = True          # set to False to skip the check (saves memory)
LEAF_MIN_SCORE = 0.40          # minimum "looks like a leaf/plant" probability
MIN_CONFIDENCE = 55.0          # below this the disease prediction is treated as unreliable
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


# ------------------------------------------------------------------
# Top bar: brand + theme switch
# ------------------------------------------------------------------
try:
    nav_left, nav_right = st.columns([3, 2], vertical_alignment="center")
except TypeError:  # older Streamlit versions
    nav_left, nav_right = st.columns([3, 2])

with nav_left:
    st.markdown(
        f'<div class="brand"><div class="mark">{LEAF_SVG}</div><b>{APP_NAME}</b></div>',
        unsafe_allow_html=True,
    )
with nav_right:
    st.radio(
        "Theme",
        ["System", "Light", "Dark"],
        key="theme",
        horizontal=True,
        format_func=lambda x: {"System": "🖥️ System", "Light": "☀️ Light", "Dark": "🌙 Dark"}[x],
        label_visibility="collapsed",
    )

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="art">{LEAF_SVG}</div>
      <h1>{APP_NAME}</h1>
      <p>Check a leaf in seconds. Upload a clear photo and the model names the most
      likely disease, with a confidence score and the closest alternatives.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
left, right = st.columns(2, gap="large")

image = None
image_bytes = None

with left:
    st.markdown(
        '<div class="upload-anchor"></div>'
        '<h3 style="margin:0 0 .2rem;font-size:1.25rem;">Upload a leaf photo</h3>'
        '<div style="color:var(--muted);font-size:.95rem;margin-bottom:1rem;line-height:1.45;">'
        "Accepts " + ", ".join(t.upper() for t in IMAGE_TYPES if t not in ("jpeg", "jfif", "tif")) + ". "
        "One leaf, in focus, on a plain background works best.</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Leaf image", type=IMAGE_TYPES, label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = load_image(image_bytes)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        except Exception:
            image = None
            image_bytes = None
            st.error("That file could not be read as an image. Try a JPG, PNG or WebP image instead.")

with right:
    if image is None:
        st.markdown(
            """
            <div class="card">
              <h3>Result</h3>
              <div class="sub">Your diagnosis will appear here.</div>
              <div class="steps">
                <div class="step"><div class="n">1</div><div><b>Upload a photo</b>
                  <span>Choose a clear picture of one leaf.</span></div></div>
                <div class="step"><div class="n">2</div><div><b>The model analyses it</b>
                  <span>It runs automatically as soon as the photo loads.</span></div></div>
                <div class="step"><div class="n">3</div><div><b>Read the result</b>
                  <span>See the likely disease, the confidence, and other possibilities.</span></div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Analysing the image…"):
                is_leaf, seen_as, leaf_score = check_leaf(image_bytes)
                results = predict(image_bytes) if is_leaf else None

            if not is_leaf:
                looks_like = f"It looks more like {html.escape(seen_as)}. " if seen_as else ""
                st.markdown(
                    f"""
                    <div class="card">
                      <h3>Result</h3>
                      <div class="sub">No diagnosis was made for this image</div>
                      <div class="verdict unsure">
                        <div class="noicon">🚫</div>
                        <div>
                          <p class="plant">Image check</p>
                          <div class="name">This doesn't look like a plant leaf</div>
                          <span class="badge">Not a leaf</span>
                        </div>
                      </div>
                      <div class="disclaimer">
                        {looks_like}Upload a close, clear photo of a single plant leaf in good light
                        and the model will check it for disease.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                top_label, top_conf = results[0]
                plant, condition = split_label(top_label)
                unsure = top_conf < MIN_CONFIDENCE
                if unsure:
                    tone, badge = "unsure", "Low confidence"
                    plant_line, condition = "Not a confident match", "No clear match"
                    list_title = f"Closest matches (low confidence)"
                else:
                    tone = "healthy" if is_healthy(condition) else "diseased"
                    badge = "Looks healthy" if tone == "healthy" else "Disease detected"
                    plant_line = f"{html.escape(plant)} leaf" if plant else "Predicted class"
                    list_title = f"Top {len(results)} predictions"

                bars = ""
                for rank, (label, conf) in enumerate(results):
                    p_name, c_name = split_label(label)
                    shown = f"{p_name} · {c_name}" if p_name else c_name
                    cls = "fill top" if rank == 0 and not unsure else "fill"
                    bars += (
                        f'<div class="bar-row"><div class="bar-head">'
                        f"<b>{html.escape(shown)}</b><span>{conf:.2f}%</span></div>"
                        f'<div class="track"><div class="{cls}" style="width:{max(conf, 0.8):.2f}%"></div></div></div>'
                    )

                st.markdown(
                    f"""
                    <div class="card">
                      <h3>Result</h3>
                      <div class="sub">Most likely class for this image</div>
                      <div class="verdict {tone}">
                        <div class="ring" style="--p:{top_conf:.1f};"><span>{top_conf:.0f}%</span></div>
                        <div>
                          <p class="plant">{plant_line}</p>
                          <div class="name">{html.escape(condition)}</div>
                          <span class="badge">{badge}</span>
                        </div>
                      </div>
                      <div class="list-title">{list_title}</div>
                      {bars}
                      <div class="disclaimer">
                        This is a model prediction, not a confirmed diagnosis. For treatment
                        decisions, check with a local agricultural expert.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if unsure:
                    st.warning("The model is not sure about this one. Try a closer, sharper photo of a single leaf in good daylight.")

        except Exception as error:
            st.error("The model could not analyse this image. Check your connection and try another photo.")
            with st.expander("Technical details"):
                st.markdown(f'<pre class="err">{html.escape(repr(error))}</pre>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tips and footer
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="tips">
      <div class="tip"><div class="emoji">☀️</div><h4>Use natural light</h4>
        <p>Daylight without harsh shadows shows the true colour of spots and lesions.</p></div>
      <div class="tip"><div class="emoji">🍃</div><h4>Frame one leaf</h4>
        <p>Fill most of the frame with a single leaf. Avoid hands, soil, and clutter.</p></div>
      <div class="tip"><div class="emoji">📷</div><h4>Keep it sharp</h4>
        <p>Hold steady and tap to focus. Blurry photos lower the model's confidence.</p></div>
    </div>
    <div class="footer">
      Built with PyTorch, Hugging Face Transformers and Streamlit<br>
      <a href="https://huggingface.co/{MODEL_ID}" target="_blank" rel="noopener">View the model on Hugging Face</a>
    </div>
    """,
    unsafe_allow_html=True,
)
