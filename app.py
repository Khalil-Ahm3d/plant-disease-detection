
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Replace with your actual Hugging Face username
MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"

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
    --sage: #E7EFE3;
    --cream: #FAF9F4;
    --ink: #26382C;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(250,249,244,0.9);
}
.block-container {
    max-width: 1160px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: var(--forest) !important;
    font-family: 'Playfair Display', serif !important;
}
.hero {
    background: linear-gradient(120deg, #183D2B, #315D40);
    color: white;
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
    color: #E0EBDD;
    font-size: 1.05rem;
    max-width: 680px;
    line-height: 1.7;
}
.eyebrow {
    color: #C5DDBF;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.section-label {
    color: #52745A;
    font-weight: 700;
    font-size: .78rem;
    letter-spacing: .12em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E9DF;
    padding: 1.1rem;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(24,61,43,.035);
}
div[data-testid="stMetricLabel"] {
    color: #617465;
}
.stButton > button {
    background: var(--forest);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: .72rem 1.3rem;
    font-weight: 700;
    min-height: 3rem;
    transition: all .2s ease;
}
.stButton > button:hover {
    background: var(--leaf);
    color: white;
    border: 0;
}
div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #A9BDA7;
    border-radius: 16px;
    padding: .6rem;
}
.result-card {
    background: white;
    border: 1px solid #E3E9DF;
    border-radius: 18px;
    padding: 1.4rem;
    margin: .6rem 0 1rem 0;
}
.result-name {
    color: #183D2B;
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 600;
    overflow-wrap: anywhere;
}
.small-note {
    color: #667568;
    font-size: .88rem;
    line-height: 1.6;
}
.footer {
    border-top: 1px solid #DFE6DB;
    padding-top: 1.2rem;
    margin-top: 3rem;
    color: #718071;
    font-size: .85rem;
    text-align: center;
}
[data-testid="stSidebar"] {
    background: #F0F3EA;
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

# ---------- Hero ----------
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
st.markdown('<div class="section-label">01 — YOUR LEAF</div>',
            unsafe_allow_html=True)
st.header("Let's examine your plant")
st.write("Choose a clear photo showing the leaf as distinctly as possible.")


uploaded_file = st.file_uploader(
    "Upload a leaf photo",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"],
    help="Supported formats: JPG, PNG, WebP, BMP, TIFF and GIF."
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.05, 1], gap="large")

        with left:
            st.image(image, caption="Your uploaded leaf", use_container_width=True)
            st.caption("For best results, use a sharp image with good lighting.")

        with right:
            st.markdown('<div class="section-label">02 — AI ANALYSIS</div>',
                        unsafe_allow_html=True)
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

                st.markdown('<div class="section-label">03 — YOUR RESULTS</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="section-label">MOST LIKELY CLASS</div>
                        <div class="result-name">{best_label}</div>
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
                        class_name = id2label.get(
                            class_id, str(class_id)
                        )
                        st.write(f"**{class_name}** — {score_percent:.1f}%")
                        st.progress(min(score_percent / 100, 1.0))

                st.info(
                    "This result is an AI prediction, not a confirmed "
                    "agricultural diagnosis. Consult a qualified expert "
                    "before making important crop-treatment decisions."
                )

    except Exception as error:
        st.error(
            "Something went wrong while loading the model or processing "
            "the image. Check the app logs for details."
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

            

    except Exception as error:
        st.error("The image could not be processed or the model could not load.")
        st.exception(error)
