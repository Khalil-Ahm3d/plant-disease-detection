# plant-disease-detection
Plant Disease Detection using Transfer Learning and MobileNetV2.

# Plant Disease Detection Using Transfer Learning

## Project Overview
This project uses transfer learning with MobileNetV2 to classify plant leaf images into disease categories. The trained model is hosted on Hugging Face and accessed through a Streamlit web application.

## Objectives
- Apply transfer learning to plant disease image classification.
- Evaluate the model on a held-out test dataset.
- Host the trained model on Hugging Face.
- Deploy a user-friendly prediction app using Streamlit and GitHub.

## Model
- Architecture: MobileNetV2
- Pretrained model: google/mobilenet_v2_1.0_224
- Number of classes: 38
- Test accuracy: 86.71%

## Technologies
- Python
- PyTorch
- Hugging Face Transformers and Hub
- Streamlit
- GitHub
- Google Colab

## How It Works
1. A user uploads a plant leaf image.
2. The image processor prepares it for the model.
3. The trained model predicts a class.
4. The application displays the predicted class and confidence score.

## Run Locally
Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Model Repository
The trained model is hosted on Hugging Face. See the `MODEL_ID` setting in `app.py`.

## Limitations
Predictions depend on the dataset and image quality. Results are not a substitute for expert agricultural diagnosis.
