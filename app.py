
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Replace YOUR_USERNAME with your Hugging Face username
MODEL_ID = "khalilahmed0062/plant-disease-detection-mobilenet"

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to predict its disease class "
    "using a transfer-learning model."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        if st.button("Detect Disease"):
            with st.spinner("Analyzing the leaf image..."):
                processor, model = load_model()
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)[0]

                predicted_id = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted_id].item()) * 100

                label_map = model.config.id2label
                disease = label_map.get(
                    predicted_id,
                    label_map.get(str(predicted_id), str(predicted_id))
                )

            st.success(f"Predicted class: {disease}")
            st.metric("Model confidence", f"{confidence:.2f}%")
            st.caption(
                "This is a model prediction, not a confirmed plant diagnosis."
            )

    except Exception as error:
        st.error("The image could not be processed or the model could not load.")
        st.exception(error)
