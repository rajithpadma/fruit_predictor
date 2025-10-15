import streamlit as st
import numpy as np
import cv2
import random
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import os

# --- Title ---
st.title("🎨 AI Drawing Accuracy Game")
st.write("Try to replicate the given image. The AI will score how close your drawing is!")

# --- Load model (pretrained feature extractor) ---
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return base_model

model = load_model()

# --- Load reference images ---
img_dir = "reference_images"
img_list = os.listdir(img_dir)
selected_image = random.choice(img_list)

st.subheader("🖼 Reference Image:")
st.image(os.path.join(img_dir, selected_image), width=300, caption="Draw this!")

# --- Upload your drawing ---
uploaded = st.file_uploader("Upload your drawing (JPG or PNG):", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    user_img = Image.open(uploaded).convert('RGB')
    st.image(user_img, width=300, caption="Your Drawing")

    # --- Preprocess both images ---
    def preprocess(img_path_or_array):
        if isinstance(img_path_or_array, str):
            img = image.load_img(img_path_or_array, target_size=(224, 224))
            img_array = image.img_to_array(img)
        else:
            img = img_path_or_array.resize((224, 224))
            img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    ref_img = preprocess(os.path.join(img_dir, selected_image))
    user_img_proc = preprocess(user_img)

    # --- Extract features ---
    ref_feat = model.predict(ref_img)
    user_feat = model.predict(user_img_proc)

    # --- Compute similarity ---
    similarity = np.dot(ref_feat, user_feat.T) / (np.linalg.norm(ref_feat) * np.linalg.norm(user_feat))
    accuracy = round(float(similarity) * 100, 2)

    st.success(f"🎯 AI Accuracy Score: **{accuracy}%**")

    if accuracy > 80:
        st.balloons()
        st.write("Excellent drawing! 🥳")
    elif accuracy > 50:
        st.write("Pretty good! Keep practicing 👏")
    else:
        st.write("Needs improvement — try again! ✏️")

st.markdown("---")
st.info("Tip: Try drawing using Paint, save it, and upload to see your AI accuracy score!")
