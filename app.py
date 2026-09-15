import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #f5f8fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #163f5f;
    margin-bottom: 5px;
}

.main-subtitle {
    text-align: center;
    font-size: 18px;
    color: #718096;
    margin-bottom: 30px;
}

/* Status */

.status {
    text-align: center;
    background-color: #eaf7ef;
    border: 1px solid #c7e8d3;
    color: #247044;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 25px;
    font-weight: 600;
}

/* Upload box */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #9db2c5;
    border-radius: 14px;
    padding: 15px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
}

/* Result cards */

.success-box {
    background-color: #eaf7ef;
    border-left: 5px solid #28a35a;
    padding: 18px;
    border-radius: 10px;
    margin: 20px 0;
}

.warning-box {
    background-color: #fff5df;
    border-left: 5px solid #e3a21a;
    padding: 18px;
    border-radius: 10px;
    margin: 20px 0;
}

.footer {
    text-align: center;
    color: #8996a5;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-Powered Brain MRI Analysis using YOLOv8'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.warning(
        "Please make sure that 'best.pt' is present "
        "in the same repository as app.py."
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.markdown(
    '<div class="status">'
    '🟢 Model Ready &nbsp; | &nbsp; YOLOv8 &nbsp; | &nbsp; '
    'Brain Tumor Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABOUT
# ============================================================

with st.container(border=True):

    st.subheader("About the Application")

    st.write(
        "This application uses a trained YOLOv8 deep learning "
        "model to analyze brain MRI images and identify "
        "possible tumor regions."
    )

    st.write(
        "Upload a brain MRI image below to generate AI-based "
        "detection results, including detected regions and "
        "confidence scores."
    )


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📤 Upload Brain MRI")

st.caption(
    "Upload a JPG, JPEG, or PNG brain MRI image for analysis."
)

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception:

        st.error("The uploaded file could not be read as an image.")
        st.stop()


    st.subheader("🖼️ MRI Preview")

    preview_col1, preview_col2, preview_col3 = st.columns(
        [1, 2, 1]
    )

    with preview_col2:

        st.image(
            image,
            caption="Uploaded Brain MRI",
            use_container_width=True
        )


    st.write("")


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🔍 Analyze MRI",
        type="primary"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze:

        with st.spinner(
            "Analyzing MRI image with YOLOv8..."
        ):

            try:

                results = model.predict(
                    source=image,
                    conf=0.4,
                    verbose=False
                )

            except Exception:

                st.error(
                    "An error occurred while analyzing the image."
                )

                st.stop()


        result = results[0]


        # ====================================================
        # DETECTION IMAGE
        # ====================================================

        plotted_image = result.plot()


        # ====================================================
        # EXTRACT DETECTIONS
        # ====================================================

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            label = result.names[class_id]

            detections.append(
                {
                    "Detected Object": label,
                    "Confidence": confidence
                }
            )


        # ====================================================
        # RESULTS
        # ====================================================

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
                plotted_image,
                channels="BGR",
                use_container_width=True
            )


        # ====================================================
        # RESULT STATUS
        # ====================================================

        if len(detections) == 0:

            st.markdown(
                """
<div class="success-box">

<b>✓ No Tumor Detected</b>

<br><br>

The model did not detect an object above the configured
confidence threshold of 40%.

</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
<div class="warning-box">

<b>⚠ Possible Tumor Detected</b>

<br><br>

The model detected one or more regions in the uploaded
MRI image.

</div>
""",
                unsafe_allow_html=True
            )


            # =================================================
            # DETECTION DETAILS
            # =================================================

            st.subheader("Detection Details")

            display_data = []

            for detection in detections:

                display_data.append(
                    {
                        "Detected Object":
                            detection["Detected Object"],

                        "Confidence":
                            f"{detection['Confidence'] * 100:.2f}%"
                    }
                )


            df = pd.DataFrame(display_data)


            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            highest_confidence = max(
                detection["Confidence"]
                for detection in detections
            )


            metric_col1, metric_col2, metric_col3 = st.columns(
                [1, 2, 1]
            )


            with metric_col2:

                st.metric(
                    "Highest Detection Confidence",
                    f"{highest_confidence * 100:.2f}%"
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.container(border=True):

    st.subheader("⚙️ How It Works")

    st.markdown(
        """
**01 — Upload**

Upload a brain MRI image in JPG, JPEG, or PNG format.

**02 — AI Analysis**

The trained YOLOv8 model analyzes the uploaded image.

**03 — Detection**

The model identifies possible tumor regions.

**04 — Results**

Bounding boxes and confidence scores are displayed for
detected regions.
"""
    )


# ============================================================
# IMPORTANT NOTICE
# ============================================================

with st.container(border=True):

    st.subheader("⚠️ Important Notice")

    st.write(
        "This application is developed for educational and "
        "research purposes only."
    )

    st.write(
        "The AI-generated results should not be considered "
        "a medical diagnosis. Always consult a qualified "
        "healthcare professional for medical interpretation."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
Brain Tumor Detection • YOLOv8 • Deep Learning • Streamlit
</div>
""",
    unsafe_allow_html=True
)
