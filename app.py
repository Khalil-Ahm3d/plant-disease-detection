
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
VALIDATOR_MODEL_ID = "openai/clip-vit-base-patch32"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# IMAGE VALIDATION CONFIGURATION
# --------------------------------------------------

IMAGE_CATEGORIES = [
    "a close-up photograph of a plant leaf",
    "a photograph of a human person",
    "a photograph of an animal",
    "a photograph of a vehicle",
    "a photograph of a building",
    "a photograph of food",
    "a photograph of a household object",
    "a photograph of a landscape or scenery",
]


class NotPlantImageError(Exception):
    """Raised when an uploaded image does not appear to show a plant leaf."""
    pass


# --------------------------------------------------
# THEME STYLING
# --------------------------------------------------

THEMES = {
    "System": {
        "background": "var(--background-color)",
        "card": "var(--secondary-background-color)",
        "text": "var(--text-color)",
        "accent": "#2e9d65",
    },
    "Light": {
        "background": "#f5f8f6",
        "card": "#ffffff",
        "text": "#17251d",
        "accent": "#238653",
    },
    "Dark": {
        "background": "#101b16",
        "card": "#1b2a22",
        "text": "#edf7ef",
        "accent": "#58c98a",
    },
}


def apply_theme(theme_name):
    theme = THEMES[theme_name]

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme["background"]};
            color: {theme["text"]};
        }}

        .main-title {{
            font-size: 2.5rem;
            font-weight: 800;
            color: {theme["accent"]};
            text-align: center;
            margin-bottom: 0.2rem;
        }}

        .subtitle {{
            text-align: center;
            font-size: 1.05rem;
            opacity: 0.8;
            margin-bottom: 2rem;
        }}

        .custom-card {{
            background: {theme["card"]};
            padding: 1.4rem;
            border-radius: 16px;
            border: 1px solid rgba(120, 160, 130, 0.2);
            margin-bottom: 1rem;
        }}

        .prediction-title {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .confidence {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {theme["accent"]};
        }}

        .footer {{
            text-align: center;
            opacity: 0.7;
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
        }}

        div.stButton > button {{
            background-color: {theme["accent"]};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            width: 100%;
        }}

        div.stButton > button:hover {{
            opacity: 0.88;
            color: white;
            border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# LOAD DISEASE DETECTION MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_disease_model():
    processor = AutoImageProcessor.from_pretrained(
        DISEASE_MODEL_ID
    )

    model = AutoModelForImageClassification.from_pretrained(
        DISEASE_MODEL_ID
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# --------------------------------------------------
# LOAD PLANT IMAGE VALIDATION MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_validator():
    processor = CLIPProcessor.from_pretrained(
        VALIDATOR_MODEL_ID
    )

    model = CLIPModel.from_pretrained(
        VALIDATOR_MODEL_ID
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# --------------------------------------------------
# CHECK WHETHER IMAGE SHOWS A PLANT LEAF
# --------------------------------------------------

def check_plant_image(image):
    processor, model = load_validator()

    inputs = processor(
        text=IMAGE_CATEGORIES,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = (
            outputs.logits_per_image.softmax(dim=1)[0]
        )

    best_index = int(torch.argmax(probabilities).item())

    best_category = IMAGE_CATEGORIES[best_index]

    best_score = float(probabilities[best_index].item())

    # Accept only images classified as plant leaves
    # with sufficient confidence.
    is_plant = (
        best_index == 0
        and best_score >= 0.30
    )

    return is_plant, best_category, best_score


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def load_image(image_bytes):
    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image


# --------------------------------------------------
# DISEASE PREDICTION
# --------------------------------------------------

def predict(image_bytes, top_k=5):

    image = load_image(image_bytes)

    # First check whether the uploaded image is a plant leaf.
    is_plant, category, score = check_plant_image(image)

    if not is_plant:
        raise NotPlantImageError(
            f"The uploaded image appears to be: {category}. "
            "Please upload a clear photograph of a plant leaf."
        )

    # Load the original disease detection model.
    processor, model = load_disease_model()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

    top_k = min(top_k, len(probabilities))

    scores, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    id2label = model.config.id2label

    predictions = []

    for score, index in zip(scores, indices):

        class_index = int(index.item())

        label = id2label.get(
            class_index,
            f"Class {class_index}",
        )

        confidence = float(score.item()) * 100

        predictions.append({
            "label": label,
            "confidence": confidence,
        })

    return predictions


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🌿 Plant Detector")

    st.write(
        "Detect plant diseases from leaf photographs "
        "using deep learning."
    )

    st.divider()

    selected_theme = st.selectbox(
        "Choose Theme",
        ["System", "Light", "Dark"],
    )

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "This application uses transfer learning "
        "with MobileNetV2 to classify plant diseases."
    )

    st.markdown("### Supported Model")

    st.caption("38 plant disease classes")

    st.caption("MobileNetV2 architecture")

    st.caption("Hugging Face model hosting")


# Apply selected theme.
apply_theme(selected_theme)


# --------------------------------------------------
# MAIN PAGE HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌱 Plant Disease Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a plant leaf image to identify possible diseases
        using an AI-powered deep learning model.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# INFORMATION CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="custom-card">
            <div class="prediction-title">🌿 Plant Validation</div>
            Check whether your uploaded image appears
            to show a plant leaf.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="custom-card">
            <div class="prediction-title">🧠 AI Detection</div>
            Use a fine-tuned MobileNetV2 model
            to classify plant diseases.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="custom-card">
            <div class="prediction-title">📊 Confidence Scores</div>
            View the top five predicted classes
            and their confidence scores.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("📸 Upload a Plant Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photograph of a plant leaf.",
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    try:
        uploaded_image = load_image(image_bytes)

        left_col, right_col = st.columns(
            [1, 1],
            gap="large",
        )

        with left_col:

            st.markdown("### Uploaded Image")

            st.image(
                uploaded_image,
                caption="Your uploaded image",
                use_container_width=True,
            )

        with right_col:

            st.markdown("### Image Information")

            st.write(
                f"**File name:** {html.escape(uploaded_file.name)}"
            )

            st.write(
                f"**Image size:** "
                f"{uploaded_image.width} × {uploaded_image.height}"
            )

            st.write(
                f"**Image format:** {uploaded_file.type}"
            )

            st.info(
                "For better results, upload a clear image "
                "where the plant leaf is easy to see."
            )

        st.divider()

        if st.button(
            "🔍 Detect Plant Disease",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Validating image and analyzing plant disease..."
                ):

                    predictions = predict(
                        image_bytes,
                        top_k=5,
                    )

                st.success(
                    "Image validation successful! Prediction completed."
                )

                # Main prediction
                best_prediction = predictions[0]

                st.markdown("## 🌱 Detection Result")

                st.markdown(
                    f"""
                    <div class="custom-card">
                        <div class="prediction-title">
                            Most Likely Prediction
                        </div>
                        <h2>{html.escape(best_prediction["label"])}</h2>
                        <div class="confidence">
                            {best_prediction["confidence"]:.2f}%
                        </div>
                        <p>Model confidence score</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top five predictions
                st.markdown("## 📊 Top 5 Predictions")

                for position, prediction in enumerate(
                    predictions,
                    start=1,
                ):

                    label = html.escape(
                        prediction["label"]
                    )

                    confidence = prediction["confidence"]

                    st.markdown(
                        f"**{position}. {label}**"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                    st.caption(
                        f"Confidence: {confidence:.2f}%"
                    )

                st.divider()

                st.info(
                    "This result is an AI prediction, not a "
                    "professional agricultural diagnosis. "
                    "For important crop decisions, consult an "
                    "agricultural expert."
                )

            except NotPlantImageError:
                st.warning(
                    "🌿 This does not appear to be a plant-leaf "
                    "image. Please upload a clear photograph "
                    "of a plant leaf."
                )

            except Exception as error:

                st.error(
                    "An error occurred while processing "
                    "your image."
                )

                with st.expander("View technical details"):
                    st.code(str(error))

    except Exception as error:

        st.error(
            "Unable to open this image. Please upload "
            "a valid JPG, JPEG, PNG, or WEBP image."
        )

        with st.expander("View technical details"):
            st.code(str(error))


else:

    st.info(
        "👆 Upload a plant leaf image to get started."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Plant Disease Detection | Transfer Learning Project
        <br>
        Powered by PyTorch, MobileNetV2, Transformers
        and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
