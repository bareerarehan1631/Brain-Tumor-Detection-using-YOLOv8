import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection | YOLOv8",
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
        background-color: #f4f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #123b5d;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 18px;
        color: #66788a;
        margin-bottom: 30px;
    }

    /* Cards */

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 650;
        color: #123b5d;
        margin-bottom: 8px;
    }

    .card-text {
        color: #64748b;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Model status */

    .status-card {
        background-color: #eaf7f0;
        border: 1px solid #c7ead7;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 25px;
        color: #17663b;
        font-weight: 600;
        text-align: center;
    }

    /* Detection result */

    .positive-result {
        background-color: #fff4e5;
        border-left: 5px solid #e39b16;
        padding: 18px;
        border-radius: 10px;
        margin: 20px 0;
    }

    .negative-result {
        background-color: #eaf7f0;
        border-left: 5px solid #24a05a;
        padding: 18px;
        border-radius: 10px;
        margin: 20px 0;
    }

    .result-title {
        font-size: 20px;
        font-weight: 700;
        color: #183b56;
    }

    .result-text {
        color: #64748b;
        margin-top: 5px;
    }

    /* Upload box */

    [data-testid="stFileUploader"] {
        background-color: white;
        border: 1px dashed #9db2c5;
        border-radius: 15px;
        padding: 15px;
    }

    /* Button */

    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #8795a5;
        font-size: 13px;
        padding-top: 25px;
        padding-bottom: 10px;
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
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()

except Exception as e:
    st.error(
        "Unable to load the trained model."
    )

    st.info(
        "Please make sure that best.pt is uploaded to the "
        "same repository folder as app.py."
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.markdown(
    """
    <div class="status-card">
        🟢 Model Ready &nbsp; | &nbsp;
        YOLOv8 &nbsp; | &nbsp;
        Brain Tumor Detection
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABOUT APPLICATION
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            About the Application
        </div>

        <div class="card-text">

            This application uses a trained YOLOv8 deep learning
            model to analyze brain MRI images and identify
            possible tumor regions.

            <br><br>

            Upload a brain MRI image below to generate
            AI-based detection results, including detected
            regions and confidence scores.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            📤 Upload Brain MRI
        </div>

        <div class="card-text">
            Select a JPG, JPEG, or PNG brain MRI image.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    st.markdown("### 🖼️ MRI Preview")

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
    # MODEL PREDICTION
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

            except Exception as e:

                st.error(
                    "An error occurred while analyzing the image."
                )

                st.stop()


        result = results[0]


        # ====================================================
        # CREATE DETECTION IMAGE
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
        # RESULTS HEADER
        # ====================================================

        st.divider()

        st.markdown("## 📊 Analysis Results")


        # ====================================================
        # ORIGINAL IMAGE / DETECTION IMAGE
        # ====================================================

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.markdown("### Original MRI")

            st.image(
                image,
                use_container_width=True
            )


        with result_col2:

            st.markdown("### YOLOv8 Detection")

            st.image(
                plotted_image,
                channels="BGR",
                use_container_width=True
            )


        # ====================================================
        # DETECTION STATUS
        # ====================================================

        if len(detections) == 0:

            st.markdown(
                """
                <div class="negative-result">

                    <div class="result-title">
                        ✓ No Tumor Detected
                    </div>

                    <div class="result-text">
                        No object was detected above the
                        configured confidence threshold of 40%.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                """
                <div class="positive-result">

                    <div class="result-title">
                        ⚠ Possible Tumor Detected
                    </div>

                    <div class="result-text">
                        The model detected one or more regions
                        in the uploaded MRI image.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # DETECTION DETAILS
            # =================================================

            st.markdown("### Detection Details")


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
            # HIGHEST CONFIDENCE
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
                    label="Highest Detection Confidence",
                    value=f"{highest_confidence * 100:.2f}%"
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            ⚙️ How It Works
        </div>

        <div class="card-text">

            <b>01 — Upload</b><br>
            Upload a brain MRI image in JPG, JPEG, or PNG format.

            <br><br>

            <b>02 — AI Analysis</b><br>
            The trained YOLOv8 model analyzes the uploaded image.

            <br><br>

            <b>03 — Detection</b><br>
            The model identifies possible tumor regions.

            <br><br>

            <b>04 — Results</b><br>
            Bounding boxes and confidence scores are displayed
            for detected regions.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMPORTANT NOTICE
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            ⚠️ Important Notice
        </div>

        <div class="card-text">

            This application is developed for educational
            and research purposes only.

            <br><br>

            The AI-generated results should not be considered
            a medical diagnosis. Always consult a qualified
            healthcare professional for medical interpretation.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Brain Tumor Detection
        &nbsp;•&nbsp;
        YOLOv8
        &nbsp;•&nbsp;
        Deep Learning
        &nbsp;•&nbsp;
        Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
