🧠 Brain Tumor Detection Using YOLOv8
Automatic detection and localization of brain tumors from MRI scans using the YOLOv8 object detection framework.

📌 Project Overview
Brain tumors are life-threatening conditions that require early and accurate diagnosis. Traditionally, doctors manually analyze MRI scans — a process that is time-consuming and subjective.

This project builds an automated brain tumor detection system that:

Detects whether a tumor is present in an MRI scan
Localizes the tumor by drawing bounding boxes around it
Classifies the tumor type (Glioma, Meningioma, Pituitary, or No Tumor)
Unlike simple classification models that only output "tumor" or "no tumor," this system also provides spatial localization, helping visualize exactly where the tumor is located.

🏷️ Tumor Classes
The model is trained to detect and classify the following four categories:

Class	Description
🔴 Glioma	Originates from glial cells. Often aggressive and grows deep inside brain tissue.
🟡 Meningioma	Develops from the meninges (protective layers). Usually benign and slow-growing.
🟣 Pituitary Tumor	Forms in the pituitary gland. Usually benign but can affect hormone levels.
🟢 No Tumor	Healthy brain MRI scans with no abnormal tumor growth (Normal class).
📊 Dataset
The dataset contains MRI brain scan images divided into Training and Testing splits, organized by tumor class:

archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
An EDA (Exploratory Data Analysis) step was performed to:

Verify the dataset is loaded correctly
Understand class-wise image distribution
Identify possible class imbalance
Ensure sufficient data for model training and evaluation
🏗️ Model Architecture
This project uses YOLOv8 (You Only Look Once — Version 8) by Ultralytics.

Why YOLOv8?

Performs detection in a single forward pass (extremely fast)
Simultaneously predicts class labels, bounding box coordinates, and confidence scores
State-of-the-art accuracy for object detection tasks
Ideal for medical imaging where both speed and accuracy are critical
Transfer Learning is applied by loading pretrained weights (yolov8n.pt) as the base model, enabling:

Faster convergence during training
Better generalization on the medical imaging dataset
⚙️ Setup & Installation
Prerequisites
Python 3.8+
Google Colab (recommended) or a local GPU environment
Install Dependencies
pip install ultralytics
Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')
Extract Dataset
!unzip "/content/Tumor_detection_MRI.v1-mri.yolov8.zip"
🚀 Training
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on the brain tumor dataset
results = model.train(data="data.yaml", epochs=50, imgsz=640)
Training Configuration:

Parameter	Value
Model	YOLOv8 Nano (yolov8n.pt)
Epochs	50
Image Size	640 × 640
Framework	Ultralytics YOLOv8
📈 Model Performance
After 50 epochs of training, the model achieved the following results across all classes:

Metric	Score
| Metric        | Score         |
| ------------- | ------------- |
| Precision (P) | 0.977 (97.7%) |
| Recall (R)    | 0.994 (99.4%) |
| mAP@0.5       | 0.993 (99.3%) |
| mAP@0.5:0.95  | 0.781 (78.1%) |

📖 Interpretation
Precision 97.7% — Very few false detections; nearly all detected tumors are real.
Recall 99.4% — Almost all actual tumors are successfully detected by the model.
mAP@0.5 99.3% — Excellent localization accuracy at a 50% bounding box overlap threshold.
mAP@0.5:0.95 78.1% — Strong performance even under stricter overlap conditions, indicating high-quality spatial localization.

# Load the best trained model
model = YOLO("/content/runs/detect/train2/weights/best.pt")

# Run inference on a test image
results = model("/path/to/test_image.jpg", conf=0.5)
results[0].show()
Loading the model from Google Drive:
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/YOLOv8_Tumor_Detection_Model/best.pt")

results = model("/path/to/test_image.jpg", conf=0.5)
results[0].show()

🤗 Hugging Face Deployment

The trained model can also be deployed using Hugging Face for real-time predictions and public demos.

Hugging Face Model Link

https://huggingface.co/spaces/Swathi-G/Brain-Tumor-Detection-YOLO

📁 Project Structure
├── Brain_Tumor_Detection_YOLOv8.ipynb   # Main Jupyter Notebook
├── data.yaml                             # YOLOv8 dataset config file
├── README.md                             # Project documentation
└── runs/
    └── detect/
        └── train2/
            └── weights/
                └── best.pt              # Best trained model weights
🛠️ Tech Stack
Tool	Purpose
YOLOv8 (Ultralytics)	Object detection framework
Python	Core programming language
OpenCV	Image loading and processing
Matplotlib	Data visualization and EDA
Google Colab	Cloud GPU training environment
Google Drive	Dataset and model storage
📜 License
This project is for educational and research purposes only. The model is not intended for clinical use without further validation.

🙏 Acknowledgements
Ultralytics YOLOv8 for the object detection framework
Brain tumor MRI dataset used for training and evaluation
Medical imaging community for open-source dataset contributions  
