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
 
st.set_page_config(
    page_title="Khalil Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');
 
:root {
  --canopy: #123B2E;
  --canopy-2: #1B5240;
  --leaf: #3F8F5B;
  --lime: #C4EB6E;
  --mist: #EDF4EE;
  --paper: #FBFDFA;
  --ink: #14261F;
  --muted: #5E7268;
  --line: #D7E4DA;
  --blight: #B5532B;
  --blight-bg: #FBEBE2;
  --ok-bg: #E4F5D3;
}
 
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', system-ui, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--mist); }
 
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }
 
h1, h2, h3, .display { font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif; letter-spacing: -0.02em; }
 
/* Hero */
.hero {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  padding: 3.2rem 3rem 3rem;
  background:
    radial-gradient(900px 380px at 88% -10%, rgba(196,235,110,.28), transparent 60%),
    linear-gradient(135deg, var(--canopy) 0%, var(--canopy-2) 100%);
  color: #F2FAEF;
  margin-bottom: 1.6rem;
}
.hero .leaf-bg {
  position: absolute; right: -30px; bottom: -50px;
  font-size: 15rem; line-height: 1; opacity: .09;
  transform: rotate(-18deg); pointer-events: none; user-select: none;
}
.hero h1 {
  font-size: clamp(2.2rem, 5vw, 3.8rem);
  font-weight: 800; line-height: 1.02; margin: 0 0 .9rem; color: #fff;
  max-width: 16ch;
}
.hero p {
  font-size: 1.08rem; line-height: 1.55; max-width: 52ch;
  color: #CFE3D2; margin: 0 0 1.4rem;
}
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip {
  padding: .38rem .85rem; border-radius: 999px; font-size: .85rem; font-weight: 500;
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18); color: #E8F5E6;
}
 
/* Cards */
.card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1.5rem 1.6rem;
}
.card h3 { font-size: 1.25rem; font-weight: 700; margin: 0 0 .25rem; }
.card .sub { color: var(--muted); font-size: .95rem; margin-bottom: 1rem; }
 
/* Uploader */
[data-testid="stFileUploader"] section {
  border: 2px dashed #A9C8B2; background: #F3F9F3; border-radius: 18px; padding: 1.4rem;
  transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--leaf); background: #EAF5EA; }
[data-testid="stFileUploader"] button {
  border-radius: 999px; border: 1px solid var(--leaf); color: var(--canopy); background: #fff; font-weight: 600;
}
 
/* Image */
[data-testid="stImage"] img { border-radius: 18px; border: 1px solid var(--line); }
 
/* Result */
.verdict {
  border-radius: 20px; padding: 1.3rem 1.4rem; display: flex; gap: 1.2rem; align-items: center;
  margin-bottom: 1.2rem;
}
.verdict.healthy { background: var(--ok-bg); border: 1px solid #BFE29A; }
.verdict.diseased { background: var(--blight-bg); border: 1px solid #EFC5AF; }
.ring {
  --p: 0; flex: none; width: 92px; height: 92px; border-radius: 50%;
  background: conic-gradient(var(--ring-color) calc(var(--p) * 1%), rgba(20,38,31,.1) 0);
  display: grid; place-items: center;
}
.ring span {
  width: 72px; height: 72px; border-radius: 50%; background: var(--paper);
  display: grid; place-items: center;
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.15rem;
}
.verdict .plant { color: var(--muted); font-size: .92rem; margin: 0; }
.verdict .name {
  font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 800;
  line-height: 1.1; margin: .1rem 0 .4rem;
}
.badge {
  display: inline-block; padding: .22rem .7rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
}
.healthy .badge { background: #2E7D32; color: #fff; }
.diseased .badge { background: var(--blight); color: #fff; }
 
/* Probability bars */
.bar-row { margin: .75rem 0; }
.bar-head { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .3rem; }
.bar-head b { font-weight: 600; }
.bar-head span { color: var(--muted); font-variant-numeric: tabular-nums; }
.track { height: 9px; border-radius: 99px; background: #E3EDE5; overflow: hidden; }
.fill { height: 100%; border-radius: 99px; background: #9DBFA9; }
.fill.top { background: linear-gradient(90deg, var(--leaf), #6BBE6F); }
 
/* Empty state */
.empty {
  text-align: center; padding: 2.6rem 1rem; color: var(--muted);
  border: 1px dashed var(--line); border-radius: 18px; background: #F6FAF6;
}
.empty .big { font-size: 2.8rem; }
.empty p { margin: .4rem auto 0; max-width: 34ch; line-height: 1.5; }
 
/* Tips */
.tips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.6rem; }
.tip { background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem; }
.tip .emoji { font-size: 1.5rem; }
.tip h4 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.02rem; margin: .4rem 0 .2rem; }
.tip p { font-size: .9rem; color: var(--muted); margin: 0; line-height: 1.45; }
@media (max-width: 800px) {
  .tips { grid-template-columns: 1fr; }
  .hero { padding: 2rem 1.4rem; }
  .verdict { flex-direction: column; text-align: center; }
}
 
.disclaimer { margin-top: 1.1rem; font-size: .85rem; color: var(--muted); line-height: 1.5; }
.footer { text-align: center; color: var(--muted); font-size: .85rem; margin-top: 2.2rem; }
 
/* Sidebar */
section[data-testid="stSidebar"] { background: var(--canopy); }
section[data-testid="stSidebar"] * { color: #E3F1E4 !important; }
 
:focus-visible { outline: 3px solid var(--lime) !important; outline-offset: 2px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
 
 
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
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
 
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
 
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
    clean = lambda s: " ".join(s.replace("_", " ").replace("(", "(").split()).strip(" ,")
    return clean(plant), clean(cond)
 
 
def is_healthy(cond: str) -> bool:
    return "healthy" in cond.lower()
 
 
# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Khalil Plant Disease Detection")
    st.write(
        "A transfer-learning model (MobileNet) fine-tuned to recognise plant "
        "leaf diseases from a single photo."
    )
    st.markdown("**Model**")
    st.code(MODEL_ID, language=None)
    st.markdown(
        f"[View on Hugging Face](https://huggingface.co/{MODEL_ID})"
    )
    show_top = st.slider("Predictions to show", 1, 5, 3)
 
# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="leaf-bg">🍃</div>
      <h1>Khalil Plant Disease Detection</h1>
      <p>Check a leaf in seconds. Upload a clear photo of a plant leaf and the model will name the most
      likely disease, with a confidence score and the runner-up possibilities.</p>
      <div class="chips">
        <span class="chip">Transfer learning</span>
        <span class="chip">MobileNet</span>
        <span class="chip">Hugging Face model</span>
        <span class="chip">Runs on Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
left, right = st.columns(2, gap="large")
 
with left:
    st.markdown(
        '<div class="card"><h3>Upload a leaf photo</h3>'
        '<div class="sub">JPG or PNG. One leaf, in focus, on a plain background works best.</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
 
    image = None
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        except Exception:
            image = None
            st.error("That file could not be read as an image. Upload a JPG or PNG and try again.")
 
with right:
    if image is None:
        st.markdown(
            """
            <div class="card">
              <h3>Result</h3>
              <div class="empty">
                <div class="big">🔍</div>
                <p>Upload a leaf photo on the left and the diagnosis will appear here.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Analysing the leaf…"):
                results = predict(image_bytes)
 
            top_label, top_conf = results[0]
            plant, condition = split_label(top_label)
            healthy = is_healthy(condition)
            tone = "healthy" if healthy else "diseased"
            ring_color = "#3F8F5B" if healthy else "#B5532B"
            badge = "Looks healthy" if healthy else "Disease detected"
            plant_line = f"{html.escape(plant)} leaf" if plant else "Predicted class"
 
            bars = ""
            for rank, (label, conf) in enumerate(results[:show_top]):
                p_name, c_name = split_label(label)
                shown = f"{p_name} · {c_name}" if p_name else c_name
                cls = "fill top" if rank == 0 else "fill"
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
                    <div class="ring" style="--p:{top_conf:.1f}; --ring-color:{ring_color};">
                      <span>{top_conf:.0f}%</span>
                    </div>
                    <div>
                      <p class="plant">{plant_line}</p>
                      <div class="name">{html.escape(condition)}</div>
                      <span class="badge">{badge}</span>
                    </div>
                  </div>
                  <b>Top {min(show_top, len(results))} predictions</b>
                  {bars}
                  <div class="disclaimer">
                    This is a model prediction, not a confirmed diagnosis. For treatment
                    decisions, check with a local agricultural expert.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            if top_conf < 60:
                st.warning(
                    "Confidence is low. Try a closer, sharper photo of a single leaf in good daylight."
                )
 
        except Exception as error:
            st.error(
                "The model could not analyse this image. Check your connection and try another photo."
            )
            with st.expander("Technical details"):
                st.exception(error)
 
# ------------------------------------------------------------------
# Tips
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="tips">
      <div class="tip"><div class="emoji">☀️</div><h4>Use natural light</h4>
        <p>Daylight without harsh shadows shows the true colour of spots and lesions.</p></div>
      <div class="tip"><div class="emoji">🍃</div><h4>Frame one leaf</h4>
        <p>Fill most of the frame with a single leaf. Avoid hands, soil, and clutter.</p></div>
      <div class="tip"><div class="emoji">📷</div><h4>Keep it sharp</h4>
        <p>Hold steady and tap to focus. Blurry photos lower the model's confidence.</p></div>
    </div>
    <div class="footer">Built with PyTorch, Hugging Face Transformers and Streamlit</div>
    """,
    unsafe_allow_html=True,
)
 
