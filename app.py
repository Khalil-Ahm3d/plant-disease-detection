
import html

import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"

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

    .stApp p, .stApp li, .stApp label {
        color: var(--ink);
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #183D2B, #315D40);
        color: #FFFFFF;
        padding: 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(24,61,43,.12);
    }

    .hero .eyebrow {
        color: #D5E8D0 !important;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.2;
        margin: .7rem 0;
    }

    .hero p {
        color: #F0F5ED !important;
        font-size: 1.05rem;
        max-width: 680px;
        line-height: 1.7;
        margin-bottom: 0;
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
        padding: 1rem;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #183D2B !important;
    }

    /* Main action button */
    .stButton > button {
        background: #183D2B !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 11px;
        min-height: 3rem;
        font-weight: 700;
    }

    .stButton > button * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background: #477A53 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        width: 100% !important;
        box-sizing: border-box !important;
        background: #FFFFFF !important;
        color: #26382C !important;
        border: 1px dashed #A9BDA7 !important;
        border-radius: 15px !important;
        padding: 14px !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: #F0F3EA !important;
        background-color: #F0F3EA !important;
        color: #26382C !important;
        border: 1px dashed #A9BDA7 !important;
        border-radius: 11px !important;
        min-height: 115px !important;
        height: auto !important;
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
        border-radius: 8px !important;
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
        border-radius: 16px;
        padding: 1.3rem;
        margin: .6rem 0 1rem;
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
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #26382C !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E3E9DF;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #183D2B !important;
    }

    /* Footer */
    .footer {
        border-top: 1px solid #DFE6DB;
        padding-top: 1.1rem;
        margin-top: 2.5rem;
        color: #526354 !important;
        font-size: .85rem;
        text-align: center;
    }

    .footer strong {
        color: #183D2B !important;
    }

    /* Mobile layout */
    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 16px;
        }

        div[data-testid="stFileUploader"] {
            padding: 9px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model ----------
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
        "An image-classification app using a fine-tuned MobileNetV2 "
        "model to predict plant disease classes."
    )

    st.markdown("### Model details")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Classification classes:** 38")
    st.write("**Test accuracy:** 86.71%")

    st.divider()
    st.caption(
        "For educational use only. Predictions on individual images "
        "may be incorrect."
    )


# ---------- Hero ----------
st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">SMARTER PLANT CARE · POWERED BY AI</div>'
    "<h1>Khalil's Plant<br>Disease Detection</h1>"
    "<p>A healthier harvest starts with understanding your leaves. "
    "Upload a plant photo to identify the most likely class.</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ---------- Overview metrics ----------
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Test accuracy", "86.71%")
metric2.metric("Disease classes", "38")
metric3.metric("Model", "MobileNetV2")

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


# ---------- Image prediction ----------
if uploaded_file is None:
    st.info(
        "👆 Upload a leaf image above to begin your plant health analysis."
    )

else:
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
                "🌱 Detect plant disease",
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

                top_count = min(3, probabilities.numel())
                scores, indices = torch.topk(
                    probabilities,
                    k=top_count,
                )

                id2label = {
                    int(key): str(value)
                    for key, value in model.config.id2label.items()
                }

                best_id = int(indices[0].item())
                best_label = id2label.get(best_id, str(best_id))
                confidence = float(scores[0].item()) * 100

            # ---------- Results ----------
            st.markdown(
                '<div class="section-label">03 — YOUR RESULTS</div>',
                unsafe_allow_html=True,
            )

            safe_label = html.escape(best_label)

            st.markdown(
                '<div class="result-card">'
                '<div class="section-label">MOST LIKELY CLASS</div>'
                f'<div class="result-name">{safe_label}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.metric(
                "Model confidence",
                f"{confidence:.1f}%",
            )

            st.caption(
                "This score reflects the model's top prediction; "
                "it is not a guarantee of correctness."
            )

            with st.expander("See the top 3 predictions"):
                for class_id_tensor, score_tensor in zip(indices, scores):
                    class_id = int(class_id_tensor.item())
                    score_percent = float(score_tensor.item()) * 100
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
                "This is an AI prediction, not a confirmed agricultural "
                "diagnosis. Consult a qualified expert before making "
                "important crop-treatment decisions."
            )

    except Exception as error:
        st.error(
            "The image could not be processed or the model could not load. "
            "Open Technical details below for the error."
        )

        with st.expander("Technical details"):
            st.exception(error)


# ---------- Footer ----------
# Keep the footer as a single-line Python string to avoid indentation errors.
st.markdown(
    '<div class="footer"><strong>Khalil\'s Plant Disease Detection</strong><br>'
    'Transfer learning · MobileNetV2 · Hugging Face · Streamlit<br>'
    'Built for learning and smarter plant care 🌱</div>',
    unsafe_allow_html=True,
)
