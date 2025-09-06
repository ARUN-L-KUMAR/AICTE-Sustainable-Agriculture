
# 🌱 AICTE Cycle 3 (2025) - Sustainable Agriculture Project  
📌 **Project: Crop Disease Detection using AI/ML**

---

## 📖 Overview
This project implements an **AI-based crop disease detection system** using **computer vision** and **deep learning**.  
The goal is to help farmers detect plant diseases early and take corrective measures to reduce crop losses and improve yield.

---

## 📂 Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Crops Covered**: Tomato, Potato, Pepper (Bell)  
- **Classes**: 15 disease categories (+ healthy leaves)  
- **Images Used**: ~7,000 processed images  

---

## ✅ Week 1 Progress (30%) – Data Processing
- ✔️ Dataset collection & organization  
- ✔️ Image preprocessing (resizing, normalization, RGB conversion)  
- ✔️ Data cleaning (invalid/corrupted image removal)  
- ✔️ Quality assurance (shape consistency, pixel value range check)  
- ✔️ Exploratory Data Analysis (EDA) with visualizations  
- ✔️ Train-test split with stratification  
- ✔️ Computed class weights for balanced training  

📊 **Outcome**: Dataset cleaned, validated, balanced, and ready for training.  

---

## ✅ Week 2 Progress (60%) – Model Development & Training
- ✔️ Implemented a **Convolutional Neural Network (CNN)**  
- ✔️ Trained baseline model for **10 epochs**  
- ✔️ Visualized training history (accuracy & loss curves)  
- ✔️ Evaluated using **classification report** & **confusion matrix**  
- ✔️ Achieved **~69% validation accuracy**  
- ✔️ Saved trained model (`crop_disease_model.h5`)  
- ✔️ Saved training history (`training_history.pkl`)  
- ✔️ Inspected misclassified samples for error analysis  

📊 **Outcome**: Working baseline model trained and evaluated. Ready for Week 3 optimization.  

---

## 🛠️ Technologies Used
- **Python 3.x**  
- **OpenCV** → Image Preprocessing  
- **NumPy / Pandas** → Data Handling  
- **Matplotlib / Seaborn** → Visualization  
- **Scikit-learn** → ML utilities (train-test split, evaluation)  
- **TensorFlow / Keras** → Deep Learning  

---

## 📂 Project Structure
```

├── week1\_data\_processing.ipynb     # Week 1: Data Processing & EDA
├── week2\_model\_training.ipynb      # Week 2: CNN Model Training & Evaluation
├── crop\_disease\_model.h5           # Saved trained model (Week 2)
├── training\_history.pkl            # Saved training history (Week 2)
├── PlantVillage/                   # Dataset (ignored in GitHub)
├── .gitignore                      # Ignore dataset & cache files
└── README.md                       # Project Documentation

````

---

## 🚀 How to Run

### Clone Repository
```bash
git clone https://github.com/ARUN-L-KUMAR/AICTE-Sustainable-Agriculture.git
cd AICTE-Sustainable-Agriculture
````

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
```

### Download Dataset

* Download PlantVillage dataset from Kaggle
* Place it inside the `PlantVillage/` folder

### Run Week 1 – Data Processing

```bash
jupyter notebook week1_data_processing.ipynb
```

### Run Week 2 – Model Training

```bash
jupyter notebook week2_model_training.ipynb
```

---

## 📅 Next Phases

* **Week 3 (60–80%) →**
  🔹 Model Optimization (Data Augmentation, Callbacks, Fine-tuning)
  🔹 Hyperparameter Tuning
  🔹 Improved accuracy & generalization

* **Week 4 (80–100%) →**
  🔹 Final Evaluation & Testing
  🔹 Model Deployment (Web/App Interface)
  🔹 Final Report & Submission

---

## 👤 Author

**Name:** ARUN KUMAR L
**Program:** AICTE Cycle 3 (2025) – Sustainable Agriculture
**Theme:** AI/ML for Agricultural Solutions

✨ *Last Updated: 06 Sept 2025*
