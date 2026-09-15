import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Brain Tumor Detection")

st.subheader(
    "AI-Powered Brain MRI Analysis using YOLOv8"
)

st.info(
    "Upload a brain MRI image to analyze it using the "
    "trained YOLOv8 detection model."
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.write(
        "Please make sure that **best.pt** is in the same "
        "folder as **app.py**."
    )

    st.stop()


st.success(
    "🟢 Model Ready — YOLOv8 Brain Tumor Detection"
)


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.header("About the Application")

st.write(
    "This application uses a trained YOLOv8 deep learning "
    "model to analyze brain MRI images and identify "
    "possible tumor regions."
)

st.write(
    "Upload an MRI image below to view the model's "
    "prediction, detected regions, and confidence scores."
)


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

st.header("📤 Upload Brain MRI")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()


    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------

    st.header("🖼️ MRI Preview")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.image(
            image,
            caption="Uploaded Brain MRI",
            use_container_width=True
        )


    # --------------------------------------------------
    # ANALYZE
    # --------------------------------------------------

    if st.button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing MRI image..."
        ):

            try:

                results = model.predict(
                    source=image,
                    conf=0.4,
                    verbose=False
                )

            except Exception as e:

                st.error(
                    "An error occurred during prediction."
                )

                st.stop()


        result = results[0]


        # --------------------------------------------------
        # DETECTION IMAGE
        # --------------------------------------------------

        detection_image = result.plot()


        # --------------------------------------------------
        # DETECTIONS
        # --------------------------------------------------

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            detections.append(
                {
                    "Detected Object": class_name,
                    "Confidence": confidence
                }
            )


        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.divider()

        st.header("📊 Analysis Results")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.subheader("Original MRI")

            st.image(
                image,
                use_container_width=True
            )


        with result_col2:

            st.subheader("YOLOv8 Detection")

            st.image(
                detection_image,
                channels="BGR",
                use_container_width=True
            )


        # --------------------------------------------------
        # NO DETECTION
        # --------------------------------------------------

        if len(detections) == 0:

            st.success(
                "✓ No Tumor Detected"
            )

            st.write(
                "The model did not detect an object above "
                "the configured confidence threshold of 40%."
            )


        # --------------------------------------------------
        # DETECTION FOUND
        # --------------------------------------------------

        else:

            st.warning(
                "⚠ Possible Tumor Detected"
            )

            st.write(
                "The model detected one or more regions "
                "in the uploaded MRI image."
            )


            st.subheader("Detection Details")


            table_data = []

            for detection in detections:

                table_data.append(
                    {
                        "Detected Object":
                            detection["Detected Object"],

                        "Confidence":
                            f"{detection['Confidence'] * 100:.2f}%"
                    }
                )


            df = pd.DataFrame(table_data)


            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            highest_confidence = max(
                detection["Confidence"]
                for detection in detections
            )


            st.metric(
                "Highest Detection Confidence",
                f"{highest_confidence * 100:.2f}%"
            )


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.divider()

st.header("⚙️ How It Works")

st.markdown(
    """
    **1. Upload:** Upload a brain MRI image.

    **2. Analyze:** The trained YOLOv8 model analyzes the image.

    **3. Detect:** The model identifies possible tumor regions.

    **4. Results:** Bounding boxes and confidence scores are displayed.
    """
)


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.header("⚠️ Important Notice")

st.warning(
    "This application is developed for educational and "
    "research purposes only. The AI-generated results "
    "should not be considered a medical diagnosis. "
    "Always consult a qualified healthcare professional "
    "for medical interpretation."
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Brain Tumor Detection • YOLOv8 • Deep Learning • Streamlit"
)
