
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
- ✔️ Evaluated using classification report & confusion matrix  
- ✔️ Achieved **~69% validation accuracy**  
- ✔️ Saved trained model (`crop_disease_model.h5`)  
- ✔️ Saved training history (`training_history.pkl`)  
- ✔️ Inspected misclassified samples for error analysis  

📊 **Outcome**: Working baseline model trained and evaluated. Ready for Week 3 optimization.  

---

## ✅ Week 3 Progress (100%) – Model Improvements & Final Submission  
- ✔️ **Data Augmentation** → Rotation, width/height shifts, shear, zoom, horizontal flip  
- ✔️ **Advanced Callbacks** → EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
- ✔️ **Extended Training** → 25 epochs with early stopping for optimal performance  
- ✔️ **Enhanced Evaluation** → Improved classification report & confusion matrix visualization  
- ✔️ **Model Optimization** → Better accuracy and robustness through regularization  
- ✔️ **Final Notebook** → Comprehensive `week3_final.ipynb` combining all weeks  
- ✔️ **Training History Visualization** → Loss, accuracy, and learning rate curves  
- ✔️ **Model Saving** → Enhanced model (`crop_disease_model_week3.h5`) and history  

📊 **Final Outcome**: Optimized crop disease detection model ready for deployment.  

---

## 🧠 Model Workflow (Week 2)  
1. **Data Loading** → Load processed dataset (Week 1 output: `processed_data.pkl`)  
2. **Preprocessing** → One-hot encode labels, normalize images (128×128, RGB)  
3. **Model Building** → CNN with:
   - Conv2D + MaxPooling + BatchNormalization (3 blocks)  
   - Dense (fully connected) layers  
   - Dropout for regularization  
   - Softmax output for multi-class classification  
4. **Training** → 10 epochs using Adam optimizer & categorical cross-entropy  
5. **Evaluation** → Accuracy/Loss plots, classification report, confusion matrix  
6. **Saving Outputs** →  
   - Trained model → `crop_disease_model.h5`  
   - Training history → `training_history.pkl`  

---

## ⚡ Note on Large Files (Git LFS)  
This repository uses **Git LFS (Large File Storage)** for model and processed data files:  
- `crop_disease_model.h5` → Trained CNN model (Week 2)  
- `processed_data.pkl` → Preprocessed dataset (Week 1 output)  
- `training_history.pkl` → Training history (Week 2)  

👉 Make sure you have [Git LFS](https://git-lfs.github.com/) installed before cloning:  
```bash
git lfs install
git clone https://github.com/ARUN-L-KUMAR/AICTE-Sustainable-Agriculture.git
````

Without Git LFS, these files will appear as small text pointers instead of the actual models.

---

## 🛠️ Technologies Used

* Python 3.x
* OpenCV → Image Preprocessing
* NumPy / Pandas → Data Handling
* Matplotlib / Seaborn → Visualization
* Scikit-learn → ML utilities (train-test split, evaluation)
* TensorFlow / Keras → Deep Learning

---

## 📂 Project Structure

```
├── week1_data_processing.ipynb     # Week 1: Data Processing & EDA
├── week2_model_training.ipynb      # Week 2: CNN Model Training & Evaluation
├── crop_disease_model.h5           # Saved trained model (via Git LFS)
├── training_history.pkl            # Saved training history (via Git LFS)
├── processed_data.pkl              # Preprocessed dataset (via Git LFS)
├── PlantVillage/                   # Dataset (ignored in GitHub)
├── .gitignore                      # Ignore dataset & cache files
├── .gitattributes                  # LFS Model & Trained Data
└── README.md                       # Project Documentation
```

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/ARUN-L-KUMAR/AICTE-Sustainable-Agriculture.git
cd AICTE-Sustainable-Agriculture
```

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

## ✅ Week 3 Progress (100%) – Model Improvements & Final Submission  
- ✔️ **Data Augmentation** → Rotation, width/height shifts, shear, zoom, horizontal flip  
- ✔️ **Advanced Callbacks** → EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
- ✔️ **Extended Training** → 25 epochs with early stopping for optimal performance  
- ✔️ **Enhanced Evaluation** → Improved classification report & confusion matrix visualization  
- ✔️ **Model Optimization** → Better accuracy and robustness through regularization  
- ✔️ **Final Notebook** → Comprehensive `week3_final.ipynb` combining all weeks  
- ✔️ **Training History Visualization** → Loss, accuracy, and learning rate curves  
- ✔️ **Model Saving** → Enhanced model (`crop_disease_model_week3.h5`) and history  

📊 **Final Outcome**: Optimized crop disease detection model ready for deployment.  

---

## 🔄 Updated Model Workflow (Week 3)  
1. **Data Loading** → Load processed dataset (Week 1 output)  
2. **Data Augmentation** → ImageDataGenerator for increased dataset diversity  
3. **Model Loading** → Load Week 2 baseline model for continued training  
4. **Advanced Training** → Extended epochs with callbacks for optimal performance  
5. **Enhanced Evaluation** → Comprehensive metrics and visualizations  
6. **Final Outputs** →  
   - Improved model → `crop_disease_model_week3.h5`  
   - Enhanced history → `training_history_week3.pkl`  
   - Final notebook → `week3_final.ipynb`  

---

## 📂 Updated Project Structure

```
├── week1_data_processing.ipynb      # Week 1: Data Processing & EDA
├── week2_model_training.ipynb       # Week 2: CNN Model Training & Evaluation
├── week3_final.ipynb               # Week 3: Final Combined Notebook (SUBMISSION)
├── crop_disease_model.h5            # Week 2 Baseline Model (via Git LFS)
├── crop_disease_model_week3.h5      # Week 3 Improved Model (via Git LFS)
├── training_history.pkl             # Week 2 Training History (via Git LFS)
├── training_history_week3.pkl       # Week 3 Training History (via Git LFS)
├── processed_data.pkl               # Preprocessed Dataset (via Git LFS)
├── PlantVillage/                    # Dataset (ignored in GitHub)
├── .gitignore                       # Ignore dataset & cache files
├── .gitattributes                   # LFS Model & Trained Data
└── README.md                        # Project Documentation
```

---

## 🎯 Week 3 Improvements Summary

### Data Augmentation
- **Rotation Range**: 20 degrees
- **Width/Height Shift**: 12% 
- **Shear Range**: 8%
- **Zoom Range**: 12%
- **Horizontal Flip**: Enabled
- **Fill Mode**: Nearest neighbor

### Advanced Callbacks
- **EarlyStopping**: Monitors validation loss, patience=5
- **ModelCheckpoint**: Saves best model based on validation loss
- **ReduceLROnPlateau**: Reduces learning rate on plateau, factor=0.5

### Training Enhancements
- **Extended Epochs**: Up to 25 epochs with early stopping
- **Batch Size**: 32 for stable training
- **Optimization**: Continued training from Week 2 baseline

---

## 🚀 Final Submission Files

### For AICTE LMS Submission:
1. **`week3_final.ipynb`** (< 10 MB) - Complete project notebook
2. **PowerPoint Presentation** (8-15 slides, < 10 MB) - Project summary

### For GitHub Repository:
- All project files including models, data, and notebooks
- Complete documentation and progress tracking

---

## 👤 Author

**Name:** ARUN KUMAR L  
**Program:** AICTE Cycle 3 (2025) – Sustainable Agriculture Internship  
**Theme:** Crop Disease Detection using AI/ML  
**GitHub:** https://github.com/ARUN-L-KUMAR/AICTE-Sustainable-Agriculture  

✨ *Final Submission: 14 Sept 2025 (Week 3 Complete)*
