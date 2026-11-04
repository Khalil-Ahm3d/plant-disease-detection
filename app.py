
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

# ---------- Premium botanical styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --forest: #183D2B;
    --leaf: #477A53;
    --cream: #FAF9F4;
    --ink: #26382C;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: var(--cream);
}

.block-container {
    max-width: 1160px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Readable headings and body text */
h1, h2, h3, h4, h5, h6 {
    color: var(--forest) !important;
    font-family: 'Playfair Display', serif !important;
}

.stApp p,
.stApp li,
.stApp label,
.stApp small {
    color: var(--ink);
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #183D2B, #315D40);
    color: #FFFFFF;
    padding: 2.6rem 2.4rem;
    border-radius: 24px;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 35px rgba(24,61,43,.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.15;
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
    background: #FFFFFF;
    border: 1px solid #E5E9DF;
    padding: 1.1rem;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(24,61,43,.035);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] * {
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

.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background: #477A53 !important;
    color: #FFFFFF !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #A9BDA7;
    border-radius: 16px;
    padding: .6rem;
}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
    color: #26382C !important;
}

/* Prediction result */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E3E9DF;
    border-radius: 18px;
    padding: 1.4rem;
    margin: .6rem 0 1rem 0;
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
    background: #F0F3EA;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #26382C !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E3E9DF;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #183D2B !important;
}

/* Alerts */
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
</style>
""", unsafe_allow_html=True)


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
        "An image-classification application that uses a "
        "fine-tuned MobileNetV2 model to predict plant disease classes."
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


# ---------- Hero section ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">SMARTER PLANT CARE · POWERED BY AI</div>
    <h1>Khalil's Plant<br>Disease Detection</h1>
    <p>
        A healthier harvest starts with understanding your leaves.
        Upload a plant photo and let the AI model identify the
        most likely class in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Overview metrics ----------
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
st.write(
    "Choose a clear photo showing the leaf as distinctly as possible."
)


# ---------- Image uploader ----------
uploaded_file = st.file_uploader(
    "Upload a leaf photo",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"],
    help="Supported formats: JPG, PNG, WebP, BMP, TIFF and GIF.",
)


# ---------- Image processing and prediction ----------
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

                inputs = processor(
                    images=image,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(
                        outputs.logits,
                        dim=-1,
                    )[0]

                count = min(3, probabilities.shape[0])
                scores, indices = torch.topk(
                    probabilities,
                    k=count,
                )

                id2label = {
                    int(key): value
                    for key, value in model.config.id2label.items()
                }

                best_id = int(indices[0].item())
                best_label = id2label.get(
                    best_id,
                    str(best_id),
                )
                confidence = float(scores[0].item()) * 100

            # ---------- Results ----------
            st.markdown(
                '<div class="section-label">03 — YOUR RESULTS</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="section-label">MOST LIKELY CLASS</div>
                    <div class="result-name">{best_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Model confidence",
                f"{confidence:.1f}%",
            )

            st.caption(
                "Confidence is the model's score for its top "
                "prediction; it is not a guarantee of correctness."
            )

            with st.expander("See the top 3 predictions"):
                for class_id, score in zip(indices, scores):
                    class_id = int(class_id.item())
                    score_percent = float(score.item()) * 100

                    class_name = id2label.get(
                        class_id,
                        str(class_id),
                    )

                    st.write(
                        f"**{class_name}** — {score_percent:.1f}%"
                    )
                    st.progress(
                        min(score_percent / 100, 1.0)
                    )

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
st.markdown("""
<div class="footer">
    <strong>Khalil's Plant Disease Detection</strong><br>
    Transfer learning · MobileNetV2 · Hugging Face · Streamlit<br>
    Built for learning and smarter plant care 🌱
</div>
""", unsafe_allow_html=True)
