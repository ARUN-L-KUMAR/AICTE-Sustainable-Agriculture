# 🌱 AICTE Cycle 3 (2025) - Sustainable Agriculture Project

## 📌 Project: Crop Disease Detection using AI/ML

### 📖 Overview

This project implements an **AI-based crop disease detection system** using computer vision and machine learning. The aim is to help farmers **detect plant diseases early** and take corrective measures to reduce crop losses and improve yield.

### 📂 Dataset

* **Source**: PlantVillage Dataset
* **Crops Covered**: Tomato, Potato, Pepper (Bell)
* **Classes**: 16 disease categories (including healthy leaves)
* **Images**: \~7,025 processed images

### ✅ Week 1 Progress (30% Completed)

* ✔️ Data collection & dataset organization
* ✔️ Image preprocessing (resizing, normalization, RGB conversion)
* ✔️ Data cleaning (invalid/corrupted image handling)
* ✔️ Quality assurance (shape consistency, pixel range checks)
* ✔️ Exploratory data analysis (EDA) with visualizations
* ✔️ Train-test split with stratification
* ✔️ Computed class weights for balanced training

📊 **Outcome:** Dataset is **clean, validated, balanced (with weights)** and ready for model training (Week 2).

### 🛠️ Technologies Used

* **Python 3.x**
* **OpenCV** → Image Processing
* **NumPy / Pandas** → Data Handling
* **Matplotlib / Seaborn** → Data Visualization
* **Scikit-learn** → ML Utilities (train-test split, class weights)

### 📂 Project Structure

```
├── week1_data_processing.ipynb   # Week 1: Data Processing & EDA
├── PlantVillage/                 # Dataset directory (ignored in GitHub)
├── week1_exports/                # Processed metadata & outputs
├── .gitignore                    # Ignore unnecessary files
└── README.md                     # Project Documentation
```

### 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone <your-repo-link>.git
   cd <repo-name>
   ```
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn opencv-python scikit-learn
   ```
3. Download the **PlantVillage dataset** and place it inside the `PlantVillage/` directory.
4. Run the notebook:

   ```bash
   jupyter notebook week1_data_processing.ipynb
   ```

### 📅 Next Phases

* **Week 2 (40–60%)** → Model Development & Initial Training
* **Week 3 (60–80%)** → Model Optimization & Validation
* **Week 4 (80–100%)** → Testing, Deployment & Documentation

### 👤 Author

**Name**: \[ARUN KUMAR L]
**Program**: AICTE Cycle 3 (2025) – Sustainable Agriculture
**Theme**: AI/ML for Agricultural Solutions

---

✨ *Last Updated: 31 Aug 2025*

---
