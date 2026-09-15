import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 10px 10px 10px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #16324F;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6f7f;
        margin-bottom: 20px;
    }

    /* Information cards */
    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e3e8ef;
        margin-bottom: 20px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.04);
    }

    .card-title {
        font-size: 20px;
        font-weight: 600;
        color: #16324F;
        margin-bottom: 8px;
    }

    .card-text {
        color: #5f6f7f;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Result cards */
    .result-success {
        background-color: #eaf8f0;
        border-left: 5px solid #20a464;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .result-warning {
        background-color: #fff7df;
        border-left: 5px solid #e5a900;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .result-title {
        font-size: 21px;
        font-weight: 700;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 14px;
        padding: 10px;
        border: 1px solid #dfe5ec;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7a8794;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">

        <div class="main-title">
            🧠 Brain Tumor Detection
        </div>

        <div class="subtitle">
            AI-Powered Brain MRI Analysis using YOLOv8
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    """
    <div class="info-card">

        <div class="card-title">
            About the Application
        </div>

        <div class="card-text">
            This application uses a trained YOLOv8 deep learning model
            to detect possible brain tumors in MRI images.
            Upload an MRI image below and the model will analyze the image
            and display the detected region along with its confidence score.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model. "
        "Please make sure 'best.pt' is present in the repository."
    )

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.markdown(
    """
    <div class="card-title">
        📤 Upload Brain MRI
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image in JPG, JPEG, or PNG format."
)


# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown(
        """
        <div class="card-title">
            🖼️ Image Preview
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    analyze = st.button(
        "🔍 Analyze MRI",
        type="primary"
    )


    if analyze:

        # =================================================
        # MODEL PREDICTION
        # =================================================

        with st.spinner("Analyzing MRI image..."):

            results = model.predict(
                source=image,
                conf=0.4
            )

        result = results[0]


        # =================================================
        # DETECTION IMAGE
        # =================================================

        plotted_image = result.plot()


        # =================================================
        # DETECTION DATA
        # =================================================

        detections = []

        for box in result.boxes:

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            label = result.names[cls_id]

            detections.append(
                {
                    "Detected Object": label,
                    "Confidence": confidence
                }
            )


        # =================================================
        # RESULT SECTION
        # =================================================

        st.divider()

        st.markdown(
            """
            <div class="card-title">
                📊 Detection Results
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # SIDE-BY-SIDE IMAGES
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Original MRI")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### AI Detection")

            st.image(
                plotted_image,
                channels="BGR",
                use_container_width=True
            )


        # =================================================
        # NO DETECTION
        # =================================================

        if len(detections) == 0:

            st.markdown(
                """
                <div class="result-warning">

                    <div class="result-title">
                        ⚠️ No Tumor Detected
                    </div>

                    <p>
                        The model did not detect an object above
                        the configured confidence threshold.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # TUMOR DETECTED
        # =================================================

        else:

            st.markdown(
                """
                <div class="result-success">

                    <div class="result-title">
                        🔎 Possible Tumor Detected
                    </div>

                    <p>
                        The model detected one or more objects
                        in the uploaded MRI image.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =============================================
            # DETECTION TABLE
            # =============================================

            display_data = []

            for detection in detections:

                display_data.append(
                    {
                        "Detected Object": detection["Detected Object"],
                        "Confidence": f"{detection['Confidence'] * 100:.2f}%"
                    }
                )


            df = pd.DataFrame(display_data)


            st.markdown("### Detection Details")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            # =============================================
            # HIGHEST CONFIDENCE
            # =============================================

            highest_confidence = max(
                detection["Confidence"]
                for detection in detections
            )

            st.metric(
                label="Highest Confidence",
                value=f"{highest_confidence * 100:.2f}%"
            )


# =========================================================
# INFORMATION SECTION
# =========================================================

st.divider()

st.markdown(
    """
    <div class="info-card">

        <div class="card-title">
            ℹ️ How It Works
        </div>

        <div class="card-text">

            <b>1.</b> Upload a brain MRI image.<br>
            <b>2.</b> The trained YOLOv8 model analyzes the image.<br>
            <b>3.</b> The model identifies detected regions.<br>
            <b>4.</b> Bounding boxes and confidence scores are displayed.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="info-card">

        <div class="card-title">
            ⚠️ Important Notice
        </div>

        <div class="card-text">

            This application is developed for educational and
            research purposes only. The results generated by
            this AI model should not be considered a medical
            diagnosis. Always consult a qualified healthcare
            professional for medical interpretation.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Brain Tumor Detection • YOLOv8 • Deep Learning • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

