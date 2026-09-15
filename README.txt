# 🧠 Brain Tumor Detection using YOLOv8

A deep learning-based **brain tumor detection system** using **YOLOv8 object detection**. The model is trained on a brain tumor MRI dataset and deployed as an interactive web application using **Streamlit**.

The application allows a user to upload a brain MRI image and receive an object-detection result with a bounding box, detected class, and confidence score.

---

## 📌 Project Overview

This project uses **Ultralytics YOLOv8** for detecting brain tumors in MRI images.

The dataset was obtained from **Roboflow** and downloaded in YOLOv8 format. The YOLOv8 Nano (`yolov8n.pt`) model was then trained on the dataset for 15 epochs.

After training, the best model weights were saved as:

```text
best.pt
```

The trained `best.pt` model is used for prediction in the Streamlit application.

**The Streamlit application does not retrain the model.**

---

## 🛠️ Technologies Used

* Python
* YOLOv8
* Ultralytics
* Roboflow
* Streamlit
* Pillow
* Pandas
* OpenCV
* Deep Learning
* Computer Vision
* Object Detection

---

## 📂 Project Structure

```text
brain-tumor-detection/
│
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

### `app.py`

The Streamlit application that:

* Loads the trained YOLOv8 model
* Allows MRI image upload
* Performs tumor detection
* Displays the detection result
* Displays bounding boxes
* Displays detected object/class
* Displays confidence scores

### `best.pt`

The trained YOLOv8 model weights generated after training.

### `requirements.txt`

Contains the Python packages required to run the Streamlit application.

### `README.md`

Provides documentation and information about the project.

---

## 🔬 Model Training

The project uses the **YOLOv8 Nano model** as the base model:

```python
model = YOLO("yolov8n.pt")
```

The model was trained using the Roboflow dataset:

```python
results = model.train(
    data=data_yaml_path,
    epochs=15,
    imgsz=640,
    batch=8,
    name="brain_tumor_detection_yolov8_roboflow"
)
```

### Training Configuration

| Parameter      | Value            |
| -------------- | ---------------- |
| Model          | YOLOv8 Nano      |
| Epochs         | 15               |
| Image Size     | 640 × 640        |
| Batch Size     | 8                |
| Task           | Object Detection |
| Dataset Format | YOLOv8           |

---

## 📊 Model Evaluation

After training, the best model weights are loaded:

```python
best_model = YOLO(best_model_path)
```

The model is then validated using the validation dataset.

The trained model is also tested on test images to evaluate its detection performance.

---

## 🔍 Prediction Workflow

The application follows this workflow:

```text
Brain MRI Image
       ↓
Upload Image
       ↓
Preprocessing
       ↓
Trained YOLOv8 Model
       ↓
Object Detection
       ↓
Tumor Detection
       ↓
Bounding Box
       ↓
Confidence Score
```

If no detection meets the configured confidence threshold, the application displays:

```text
No tumor detected.
```

---

## 🌐 Streamlit Application

The trained model is deployed using **Streamlit**.

The application allows users to:

1. Upload a brain MRI image.
2. View the uploaded image.
3. Run tumor detection.
4. View the detected tumor with a bounding box.
5. View the detected object/class.
6. View the confidence score.

The trained model is loaded directly from:

```text
best.pt
```

Therefore, the model does **not need to be trained again every time the Streamlit application runs**.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/brain-tumor-detection.git
```

Navigate to the project directory:

```bash
cd brain-tumor-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Run:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## ☁️ Streamlit Deployment

This project can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Upload the following files to your GitHub repository:

```text
app.py
requirements.txt
best.pt
README.md
```

2. Connect your GitHub account to Streamlit Community Cloud.
3. Select the repository.
4. Select the `main` branch.
5. Select:

```text
app.py
```

as the main application file.
6. Click **Deploy**.

Streamlit will install the dependencies from `requirements.txt` and run `app.py`.

---

## 🔐 Security

The Roboflow API key used during dataset download is **not required by the deployed Streamlit application**.

API keys and other private credentials should never be uploaded to GitHub.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The predictions generated by this application should **not be considered a medical diagnosis**. Brain MRI results should be evaluated and interpreted by qualified medical professionals.

---

## 👩‍💻 Author

**Bareera Rehan**

Data Science & AI Student

### Areas of Interest

* Data Science
* Machine Learning
* Deep Learning
* Generative AI
* Computer Vision
* Artificial Intelligence
