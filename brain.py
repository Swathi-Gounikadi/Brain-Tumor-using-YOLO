import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection using YOLOv8")
st.write("Upload an MRI image to detect tumors.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Tumor"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model(temp_path, conf=0.5)
        st.image(results[0].plot(), caption="Detection Result", use_container_width=True)

        os.remove(temp_path)