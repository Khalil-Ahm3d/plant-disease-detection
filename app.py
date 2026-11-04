import html

import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------- Model configuration ----------
MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Khalil's Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --forest: #183D2B;
    --leaf: #477A53;
    --cream: #FAF9F4;
    --ink: #26382C;
    --sage: #F0F3EA;
}

.stApp {
    background: var(--cream) !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: var(--cream) !important;
}

.block-container {
    max-width: 1160px;
    padding: 2rem 2rem 3rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--forest) !important;
    font-family: 'Playfair Display', serif !important;
}

.stApp p, .stApp li, .stApp label, .stApp small {
    color: var(--ink);
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #183D2B, #315D40);
    color: #FFFFFF;
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 35px rgba(24,61,43,.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.2;
    margin-bottom: .8rem;
}

.hero p {
    color: #F0F5ED !important;
    font-size: 1.05rem;
    max-width: 680px;
    line-height: 1.7;
}

.eyebrow {
    color: #D5E8D0 !important;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.section-label {
    color: #315D40 !important;
    font-weight: 700;
    font-size: .78rem;
    letter-spacing: .12em;
    text-transform: uppercase;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E9DF;
    padding: 1.1rem;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(24,61,43,.035);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"] {
    color: #183D2B !important;
}

/* Buttons */
.stButton > button {
    background: #183D2B !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 12px;
    padding: .72rem 1.3rem;
    font-weight: 700;
    min-height: 3rem;
}

.stButton > button *,
.stButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background: #477A53 !important;
}

/* Uploader: light drop zone with readable text */
div[data-testid="stFileUploader"] {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    color: #26382C !important;
    border: 1px dashed #A9BDA7 !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #F0F3EA !important;
    background-color: #F0F3EA !important;
    color: #26382C !important;
    border: 1px dashed #A9BDA7 !important;
    border-radius: 12px !important;
    min-height: 120px !important;
    box-sizing: border-box !important;
}

div[data-testid="stFileUploaderDropzone"] *,
div[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #26382C !important;
    opacity: 1 !important;
}

div[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    color: #183D2B !important;
    border: 1px solid #A9BDA7 !important;
    border-radius: 9px !important;
}

div[data-testid="stFileUploader"] button * {
    color: #183D2B !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #E7EFE3 !important;
}

/* Result card */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E3E9DF;
    border-radius: 18px;
    padding: 1.4rem;
    margin: .6rem 0 1rem;
}

.result-card .section-label {
    color: #315D40 !important;
}

.result-name {
    color: #183D2B !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 600;
    overflow-wrap: anywhere;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F0F3EA !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #26382C !important;
}

/* Expanders and alerts */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E3E9DF;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #183D2B !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #26382C !important;
}

/* Footer */
.footer {
    border-top: 1px solid #DFE6DB;
    padding-top: 1.2rem;
    margin-top: 3rem;
    color: #526354 !important;
    font-size: .85rem;
    text-align: center;
}

.footer strong {
    color: #183D2B !important;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
    }
    div[data-testid="stFileUploader"] {
        padding: 10px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 Khalil's")
    st.caption("AI · AGRICULTURE · PLANT HEALTH")
    st.divider()

    st.markdown("### About this project")
    st.write(
        "An image-classification application that uses a fine-tuned "
        "MobileNetV2 model to predict plant disease classes."
    )

    st.markdown("### Model details")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Classification classes:** 38")
    st.write("**Test accuracy:** 86.71%")

    st.divider()
    st.caption(
        "Predictions are intended for educational use. "
        "Accuracy on an individual image may vary."
    )


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
    <div class="eyebrow">SMARTER PLANT CARE · POWERED BY AI</div>
    <h1>Khalil's Plant<br>Disease Detection</h1>
    <p>
        A healthier harvest starts with understanding your leaves.
        Upload a plant photo and let the AI model identify the
        most likely class in seconds.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Overview ----------
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", "86.71%")
m2.metric("Disease classes", "38")
m3.metric("Model", "MobileNetV2")

st.write("")
st.markdown(
    '<div class="section-label">01 — YOUR LEAF</div>',
    unsafe_allow_html=True,
)
st.header("Let's examine your plant")
st.write("Choose a clear photo showing the leaf as distinctly as possible.")


# ---------- Image uploader ----------
uploaded_file = st.file_uploader(
    "Upload a leaf photo",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"],
    help="Supported formats: JPG, PNG, WebP, BMP, TIFF and GIF.",
)


# ---------- Prediction ----------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.05, 1], gap="large")

        with left:
            st.image(
                image,
                caption="Your uploaded leaf",
                use_container_width=True,
            )
            st.caption(
                "For best results, use a sharp image with good lighting."
            )

        with right:
            st.markdown(
                '<div class="section-label">02 — AI ANALYSIS</div>',
                unsafe_allow_html=True,
            )
            st.subheader("Ready to inspect")
            st.write(
                "The model will compare your image with patterns "
                "learned during training."
            )
            detect = st.button(
                "🌱  Detect plant disease",
                use_container_width=True,
            )

        if detect:
            with st.spinner("Analyzing your leaf..."):
                processor, model = load_model()
                inputs = processor(images=image, return_tensors="pt")

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(
                        outputs.logits, dim=-1
                    )[0]

                count = min(3, probabilities.shape[0])
                scores, indices = torch.topk(probabilities, k=count)

                id2label = {
                    int(key): value
                    for key, value in model.config.id2label.items()
                }

                best_id = int(indices[0].item())
                best_label = id2label.get(best_id, str(best_id))
                confidence = float(scores[0].item()) * 100

            st.markdown(
                '<div class="section-label">03 — YOUR RESULTS</div>',
                unsafe_allow_html=True,
            )
            safe_label = html.escape(str(best_label))
            st.markdown(
                f"""
<div class="result-card">
    <div class="section-label">MOST LIKELY CLASS</div>
    <div class="result-name">{safe_label}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.metric("Model confidence", f"{confidence:.1f}%")
            st.caption(
                "Confidence is the model's score for its top prediction; "
                "it is not a guarantee of correctness."
            )

            with st.expander("See the top 3 predictions"):
                for class_id, score in zip(indices, scores):
                    class_id = int(class_id.item())
                    score_percent = float(score.item()) * 100
                    class_name = id2label.get(class_id, str(class_id))
                    st.write(f"**{class_name}** — {score_percent:.1f}%")
                    st.progress(min(score_percent / 100, 1.0))

            st.info(
                "This result is an AI prediction, not a confirmed "
                "agricultural diagnosis. Consult a qualified expert "
                "before making important crop-treatment decisions."
            )

    except Exception as error:
        st.error(
            "Something went wrong while loading the model or "
            "processing the image. Check the app logs for details."
        )
        with st.expander("Technical details"):
            st.exception(error)

else:
    st.info(
        "👆 Upload a leaf image above to begin your plant health analysis."
    )


# ---------- Footer ----------
st.markdown(
    """
<div class="footer">
    <strong>Khalil's Plant Disease Detection</strong><br>
    Transfer learning · MobileNetV2 · Hugging Face · Streamlit<br>
    Built for learning and smarter plant care 🌱
</div>
""",
    unsafe_allow_html=True,
)

    Transfer learning · MobileNetV2 · Hugging Face · Streamlit<br>
    Built for learning and smarter plant care 🌱
</div>
""", unsafe_allow_html=True)
