import streamlit as st
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ===============================
# PATHS
# ===============================
MODEL_PATH = r"C:\Users\khaiw\OneDrive\Desktop\vs code python\GarbageClassifier7.keras"
CLASS_INDICES_PATH = "class_indices.json"

# ===============================
# LOAD MODEL & CLASSES
# ===============================
model = load_model(MODEL_PATH, compile=False)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

CATEGORIES = {v: k for k, v in class_indices.items()}

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_waste(image):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img, verbose=0)
    category_idx = int(np.argmax(preds))
    category = CATEGORIES[category_idx]
    confidence = round(100 * np.max(preds), 2)
    return category, confidence

# ===============================
# RECYCLING TIPS
# ===============================
RECYCLING_TIPS = {
    "plastic": "💡 Tip: Clean and separate plastics before recycling.",
    "metal": "💡 Tip: Rinse metals and remove non-metal parts.",
    "glass": "💡 Tip: Avoid broken glass; recycle in designated bins.",
    "cardboard": "💡 Tip: Flatten cardboard boxes before recycling.",
    "paper": "💡 Tip: Keep paper dry and free from food stains.",
    "trash": "💡 Tip: Dispose of non-recyclables properly to reduce contamination."
}

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻️ Garbage Classifier with Recycling Tips")
st.write("Upload an image OR use your webcam to classify waste.")

option = st.radio("Choose input method:", ("Upload Image", "Webcam"))

# ----------- Upload Image -----------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        st.image(img_array, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Waste"):
            category, confidence = predict_waste(img_array)
            tip = RECYCLING_TIPS.get(category, "")
            st.success(f"Prediction: **{category}** ({confidence}%)")
            st.info(tip)

# ----------- Webcam Input -----------
elif option == "Webcam":
    stframe = st.empty()
    run = st.checkbox("Start Webcam")
    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        stframe.warning("Live preview running. Click 'Capture & Predict' to classify one frame.")
        
        # Create a "Predict" button
        predict_button = st.button("Capture & Predict")
        frame_captured = False

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access webcam")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)

            # Only predict once when button clicked
            if predict_button and not frame_captured:
                category, confidence = predict_waste(rgb_frame)
                tip = RECYCLING_TIPS.get(category, "")
                st.success(f"Prediction: **{category}** ({confidence}%)")
                st.info(tip)
                frame_captured = True  # prevents flooding

        cap.release()
    else:
        stframe.empty()
