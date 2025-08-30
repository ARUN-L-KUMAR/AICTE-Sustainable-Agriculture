# 🌱 Crop Disease Detection - Week 1 (Data Processing)

This repository contains the **Week 1 work** for a Crop Disease Detection project using the **PlantVillage Dataset**.  
The focus here is on **data preprocessing** — preparing raw leaf images into a structured dataset suitable for deep learning models.  

---

## 📌 Project Overview
The **PlantVillage dataset** contains images of healthy and diseased crop leaves.  
The goal of this project is to:
- Preprocess images for machine learning.
- Organize data into training and testing sets.
- Visualize class distribution and sample images.

This is the **first step (Week 1)** of building a complete **Crop Disease Detection pipeline**.

---

## 📂 Dataset
- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- The dataset consists of **54,000+ images** of healthy and diseased plant leaves.  
- Each folder in the dataset represents a **class (crop + condition)**.  

> ⚠️ **Note:** Due to large size, the dataset is **not included in this repository**.  
Please download it from Kaggle and place it in the project directory as:

/PlantVillage
├── Apple___Black_rot
├── Apple___healthy
├── Corn___Cercospora_leaf_spot
├── Corn___healthy
└── ... etc.


---

## ⚙️ Steps Implemented (Week 1)

1. **Import Libraries**  
   - `numpy`, `pandas`, `cv2`, `matplotlib`, `seaborn`, `scikit-learn`

2. **Load Dataset**  
   - Read images from folders  
   - Resize to `128x128`  
   - Normalize pixel values (0-1)  

3. **Create Labels**  
   - Each folder is mapped to a **class index**

4. **Split Dataset**  
   - Training: 80%  
   - Testing: 20%  
   - Stratified split ensures balanced classes  

5. **Visualizations**  
   - Class distribution using `seaborn`  
   - Sample training images preview  

---

## 📊 Example Outputs

### ✅ Class Distribution
Bar chart showing how many images exist per class.

### ✅ Sample Images
Grid of training samples with labels for quick verification.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

    Install dependencies:

pip install -r requirements.txt

Download dataset from Kaggle and place in the project folder:

/PlantVillage/

Run the script:

    python week1_data_processing.py

📦 Requirements

    Python 3.8+

    numpy

    pandas

    matplotlib

    seaborn

    opencv-python

    scikit-learn

Install with:

pip install -r requirements.txt

📅 Weekly Progress

    Week 1: Data Processing (✅ Current stage)

    Week 2: Model Building (CNN, Transfer Learning)

    Week 3: Model Training & Evaluation

    Week 4: Deployment & Inference

✨ Future Work

    Train Convolutional Neural Networks (CNNs)

    Implement Transfer Learning (ResNet, VGG, EfficientNet)

    Build Web or Mobile app for farmers to upload leaf images and detect diseases

📌 Acknowledgments

    Dataset: PlantVillage (Kaggle)

Inspiration: Agricultural AI solutions for smart farming
