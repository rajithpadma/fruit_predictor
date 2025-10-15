import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import os
from skimage.metrics import structural_similarity as ssim
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Drawing Judge", page_icon="🎨", layout="centered")

st.title("🎨 AI Drawing Judge Game")
st.write("Replicate the fruit shown below! Draw it using your mouse — the AI will score your accuracy 🍎")

# List all image files in the same folder as this script
image_extensions = [".jpg", ".jpeg", ".png"]
available_images = [f for f in os.listdir() if os.path.splitext(f)[1].lower() in image_extensions]

if not available_images:
    st.error("⚠️ No image files found in the folder! Please add some fruit images (jpg/png).")
    st.stop()

# Pick a random fruit image for each session
if "target_image" not in st.session_state:
    st.session_state.target_image = random.choice(available_images)

# Option to switch to a new fruit
if st.button("🔄 New Fruit"):
    st.session_state.target_image = random.choice(available_images)

# Load target image
target_path = st.session_state.target_image
target_img = Image.open(target_path).convert("RGB")

st.image(target_img, caption=f"🖼️ Draw this: {os.path.splitext(st.session_state.target_image)[0].capitalize()}", use_container_width=True)

# Canvas for drawing
st.write("✏️ Draw your version below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # White background
    stroke_width=6,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Evaluate the drawing when user clicks the button
if st.button("🏁 Submit Drawing"):
    if canvas_result.image_data is not None:
        # Convert canvas to PIL Image
        user_img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"))

        # Resize both to same dimensions
        target_resized = target_img.resize((224, 224))
        user_resized = user_img.resize((224, 224))

        # Convert to grayscale for comparison
        img1 = np.array(target_resized.convert("L"))
        img2 = np.array(user_resized.convert("L"))

        # Compute structural similarity (SSIM)
        score, diff = ssim(img1, img2, full=True)
        st.subheader(f"🧠 AI Accuracy Score: {score * 100:.2f}%")

        # Feedback
        if score > 0.85:
            st.success("Amazing! That looks just like it 🎯")
        elif score > 0.6:
            st.info("Nice work! It’s quite similar 🍏")
        else:
            st.warning("Keep practicing! Try to match the shape better 🍌")
    else:
        st.warning("Please draw something before submitting!")
