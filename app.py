
import html

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"

st.set_page_config(
    page_title="Khalil's Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN: FIXED BOTANICAL COLORS FOR BOTH THEMES
# =========================================================

st.markdown(
    """
<style>
/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #FAF9F4 !important;
    color: #26382C !important;
}

[data-testid="stHeader"] {
    background: #FAF9F4 !important;
}

.block-container {
    max-width: 1160px !important;
    padding: 2rem 2rem 3rem !important;
}

/* General headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #183D2B !important;
}

/* General text */
.stApp p,
.stApp li,
.stApp label {
    color: #26382C;
}

/* =========================================================
   HERO BANNER AND MAIN TITLE
   White title on a dark green background
   ========================================================= */

div.hero {
    background: linear-gradient(125deg, #123522 0%, #285B3B 100%) !important;
    padding: 2.5rem !important;
    border-radius: 22px !important;
    margin-bottom: 1.6rem !important;
    box-shadow: 0 10px 28px rgba(24, 61, 43, 0.16) !important;
    color: #FFFFFF !important;
}

div.hero .eyebrow {
    color: #E0EEDF !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    opacity: 1 !important;
}

/* Strong title styling to override global heading rules */
div.hero h1,
div.hero h1 *,
div.hero h1 a {
    color: #FFFFFF !important;
    background: transparent !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: clamp(2.2rem, 5vw, 3.4rem) !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
    opacity: 1 !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.18) !important;
    margin: 0.7rem 0 !important;
}

div.hero p,
div.hero p * {
    color: #F3F7F0 !important;
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
    max-width: 700px;
    opacity: 1 !important;
    background: transparent !important;
    margin-bottom: 0 !important;
}

/* Section labels */
.section-label {
    color: #315D40 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE6D9 !important;
    border-radius: 15px !important;
    padding: 1rem !important;
}

div[data-testid="stMetric"] *,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: #183D2B !important;
}

/* =========================================================
   UPLOADER: FIXED LIGHT BACKGROUND AND DARK TEXT
   ========================================================= */

div[data-testid="stFileUploader"] {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    color: #26382C !important;
    border: 1px solid #DDE6D9 !important;
    border-radius: 15px !important;
    padding: 14px !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #F0F3EA !important;
    background-color: #F0F3EA !important;
    color: #26382C !important;
    border: 1px dashed #8DA88A !important;
    border-radius: 12px !important;
    min-height: 120px !important;
    height: auto !important;
    box-sizing: border-box !important;
}

/* Keep uploader instructions and file limit readable */
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] [data-testid],
div[data-testid="stFileUploaderDropzone"] *,
div[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #26382C !important;
    opacity: 1 !important;
}

/* Browse files button */
div[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    color: #183D2B !important;
    border: 1px solid #9BB49A !important;
    border-radius: 9px !important;
}

div[data-testid="stFileUploader"] button *,
div[data-testid="stFileUploader"] button span {
    color: #183D2B !important;
    fill: #183D2B !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #E4EDE0 !important;
}

/* Main action button */
.stButton > button {
    background: #183D2B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 11px !important;
    min-height: 3rem !important;
    font-weight: 700 !important;
}

.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background: #386746 !important;
}

/* Prediction result card */
.result-card {
    background: #FFFFFF !important;
    border: 1px solid #DDE6D9 !important;
    border-radius: 16px !important;
    padding: 1.3rem !important;
    margin: 0.6rem 0 1rem !important;
}

.result-card .section-label {
    color: #315D40 !important;
}

.result-name {
    color: #183D2B !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    overflow-wrap: anywhere !important;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #F0F3EA !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {
    color: #26382C !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE6D9 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #183D2B !important;
}

/* Footer */
.footer {
    border-top: 1px solid #DDE6D9;
    padding-top: 1.2rem;
    margin-top: 2.5rem;
    text-align: center;
    color: #526354 !important;
    font-size: 0.85rem;
}

.footer strong {
    color: #183D2B !important;
}

/* Mobile layout */
@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 2rem !important;
    }

    div.hero {
        padding: 1.5rem !important;
        border-radius: 16px !important;
    }

    div.hero h1 {
        font-size: 2.2rem !important;
    }

    div[data-testid="stFileUploader"] {
        padding: 8px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD THE HUGGING FACE MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🌿 Khalil's")
    st.caption("AI · AGRICULTURE · PLANT HEALTH")
    st.divider()

    st.markdown("### About this project")
    st.write(
        "This application uses a fine-tuned MobileNetV2 image "
        "classification model to predict plant disease classes "
        "from leaf photographs."
    )

    st.markdown("### Model information")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Classification classes:** 38")
    st.write("**Test accuracy:** 86.71%")

    st.divider()
    st.caption(
        "For educational use. Predictions may be incorrect and "
        "should not replace professional agricultural advice."
    )


# =========================================================
# HERO BANNER
# =========================================================

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">SMARTER PLANT CARE · POWERED BY AI</div>
    <h1>Khalil's Plant<br>Disease Detection</h1>
    <p>
        A healthier harvest starts with understanding your leaves.
        Upload a clear plant photo and let the model predict the
        most likely plant disease class.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# OVERVIEW METRICS
# =========================================================

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

st.write(
    "Upload a clear image of a leaf. Good lighting and a focused "
    "image can help the model make a better prediction."
)


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a leaf photo",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "gif",
    ],
    help=(
        "Supported formats: JPG, JPEG, PNG, WebP, BMP, TIFF and GIF."
    ),
)


# =========================================================
# IMAGE PREDICTION
# =========================================================

if uploaded_file is None:
    st.info(
        "Upload a leaf image above to begin your plant health analysis."
    )

else:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        image_col, analysis_col = st.columns([1, 1], gap="large")

        with image_col:
            st.image(
                image,
                caption="Your uploaded leaf",
                use_container_width=True,
            )
            st.caption(
                "Tip: use a sharp photo with the leaf clearly visible."
            )

        with analysis_col:
            st.markdown(
                '<div class="section-label">02 — AI ANALYSIS</div>',
                unsafe_allow_html=True,
            )

            st.subheader("Ready to inspect")

            st.write(
                "Start the analysis to compare your photo with patterns "
                "learned by the model during training."
            )

            detect_button = st.button(
                "🌱 Detect plant disease",
                use_container_width=True,
            )

        if detect_button:
            with st.spinner("Loading the model and analyzing your image..."):
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

                predicted_id = int(indices[0].item())

                predicted_label = id2label.get(
                    predicted_id,
                    str(predicted_id),
                )

                confidence = float(scores[0].item()) * 100

            st.markdown(
                '<div class="section-label">03 — YOUR RESULTS</div>',
                unsafe_allow_html=True,
            )

            safe_label = html.escape(predicted_label)

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
                "Confidence is the model's score for its top prediction. "
                "It does not guarantee that the prediction is correct."
            )

            with st.expander("View the top 3 predictions"):
                for class_tensor, score_tensor in zip(indices, scores):
                    class_id = int(class_tensor.item())
                    score_percent = float(score_tensor.item()) * 100

                    class_name = id2label.get(
                        class_id,
                        str(class_id),
                    )

                    st.write(
                        f"**{class_name}** — {score_percent:.1f}%"
                    )

                    st.progress(
                        min(max(score_percent / 100, 0.0), 1.0)
                    )

            st.info(
                "This is an AI prediction, not a confirmed diagnosis. "
                "Consult a qualified agricultural expert before making "
                "important crop-treatment decisions."
            )

    except Exception as error:
        st.error(
            "The image could not be processed or the model could not load."
        )

        with st.expander("Technical details"):
            st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    "<strong>Khalil's Plant Disease Detection</strong><br>"
    "Transfer learning · MobileNetV2 · Hugging Face · Streamlit<br>"
    "Built for learning and smarter plant care 🌱"
    "</div>",
    unsafe_allow_html=True,
)
