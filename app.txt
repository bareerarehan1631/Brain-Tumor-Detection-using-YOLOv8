import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image to detect a tumor.")

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded MRI")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Tumor"):

        with st.spinner("Analyzing image..."):

            results = model.predict(
                source=image,
                conf=0.4
            )

        result = results[0]

        # Show detection result
        plotted_image = result.plot()

        st.subheader("Detection Result")
        st.image(plotted_image, channels="BGR", use_container_width=True)

        detections = []

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[cls_id]

            detections.append({
                "Detected Object": label,
                "Confidence Score": round(confidence, 3)
            })

        if len(detections) == 0:

            st.warning("No tumor detected.")

        else:

            st.success("Tumor detected.")

            df = pd.DataFrame(detections)

            st.subheader("Detection Details")
            st.dataframe(df, use_container_width=True)