
# 🌱 AICTE Cycle 3 (2025) - Sustainable Agriculture Project  

📌 **Project: Crop Disease Detection using AI/ML**  

---

## 📖 Overview  
This project implements an **AI-based crop disease detection system** using **computer vision** and **deep learning**.  
The goal is to help farmers detect plant diseases early and take corrective measures to reduce crop losses and improve yield.  

**🎯 Project Results:**
- **Final Model Accuracy**: 63.4% (Week 3 Enhanced Model)
- **Baseline Accuracy**: 80.9% (Week 2 Model)
- **Total Images Processed**: 7,025 images
- **Training Strategy**: Data augmentation with advanced callbacks

---

## 📂 Dataset  
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Crops Covered**: Tomato, Potato, Pepper (Bell)  
- **Classes**: 15 disease categories (+ healthy leaves)  
- **Total Images**: 7,025 processed images
- **Image Resolution**: 128×128 pixels (RGB)  

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
- ✔️ Achieved **80.9% validation accuracy** (baseline model)
- ✔️ Saved trained model (`crop_disease_model.h5`)  
- ✔️ Saved training history (`training_history.pkl`)  
- ✔️ Inspected misclassified samples for error analysis  

📊 **Outcome**: Strong baseline model with 80.9% accuracy. Ready for Week 3 optimization.  

---

## ✅ Week 3 Progress (100%) – Model Improvements & Final Submission  
- ✔️ **Data Augmentation** → Rotation, width/height shifts, shear, zoom, horizontal flip  
- ✔️ **Advanced Callbacks** → EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
- ✔️ **Extended Training** → 25 epochs with early stopping (stopped at epoch 10)
- ✔️ **Enhanced Evaluation** → Detailed classification report & confusion matrix  
- ✔️ **Model Analysis** → Comprehensive diagnostic analysis of performance differences
- ✔️ **Final Notebook** → Complete `final_crop_disease_detection.ipynb` (all 3 weeks)
- ✔️ **Training History Visualization** → Loss, accuracy, and learning rate curves  
- ✔️ **Performance Results** → Final accuracy: 63.4% (vs 80.9% baseline)

📊 **Final Outcome**: Enhanced model with data augmentation. Performance analysis shows overfitting challenges with augmented data.

---

## 📊 Final Project Results & Analysis

### Model Performance Comparison
| Metric | Week 2 Baseline | Week 3 Enhanced | Difference |
|--------|----------------|-----------------|------------|
| **Validation Accuracy** | 80.9% | 63.4% | -17.5% |
| **Training Epochs** | 10 epochs | 10 epochs (early stopping) | Same |
| **Dataset Size** | 7,025 images | 7,025 + augmentation | Increased |
| **Training Strategy** | Standard training | Data augmentation + callbacks | Enhanced |

### Key Findings
1. **Baseline Model**: Strong performance (80.9%) with standard training approach
2. **Enhanced Model**: Lower accuracy (63.4%) despite data augmentation
3. **Root Cause**: Overfitting due to aggressive data augmentation on limited dataset
4. **Learning**: Data augmentation requires careful tuning for small datasets

### Technical Insights
- **Data Augmentation Impact**: Heavy augmentation (rotation, shifts, zoom) may have made training too challenging
- **Early Stopping**: Model stopped at epoch 10, suggesting optimization difficulties
- **Class Imbalance**: Some disease classes had fewer samples, affecting augmentation effectiveness
- **Model Architecture**: CNN architecture may need adjustment for augmented data

### Future Improvements
1. **Reduce Augmentation Intensity**: Lower rotation angles, smaller shift ranges
2. **Progressive Training**: Start with less augmentation, gradually increase
3. **Architecture Optimization**: Add more regularization layers
4. **Dataset Expansion**: Collect more real-world samples for each class

---## 🧠 Model Workflow (Week 2)  
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
├── data/
│   ├── PlantVillage/                   # Raw images dataset (ignored in GitHub)
│   └── processed_data.pkl              # Preprocessed dataset (via Git LFS)
├── models/
│   ├── best_model_week3.h5             # Best checkpoint for week 3 (via Git LFS)
│   ├── crop_disease_model.h5           # Week 2 model (via Git LFS)
│   ├── crop_disease_model_week2.h5     # Week 2 model (via Git LFS)
│   ├── crop_disease_model_week3.h5     # Week 3 model (via Git LFS)
│   └── crop_disease_model_week3.keras  # Week 3 model in Keras format (via Git LFS)
├── notebooks/
│   ├── week1_data_processing.ipynb     # Week 1: Data Processing & EDA
│   ├── week2_model_training.ipynb      # Week 2: CNN Model Training & Evaluation
│   ├── week3.ipynb                     # Week 3: Development & Exploration
│   ├── week3_final.ipynb               # Week 3: Final Model Training & Evaluation
│   └── final_crop_disease_detection.ipynb # Week 3: Final Combined Notebook (SUBMISSION)
├── results/
│   ├── training_history.pkl            # Saved training history (via Git LFS)
│   ├── training_history_week2.pkl      # Week 2 training history (via Git LFS)
│   └── training_history_week3.pkl      # Week 3 training history (via Git LFS)
├── .gitignore                          # Ignore dataset & cache files
├── .gitattributes                      # LFS Model & Trained Data
├── Week3_PPT_Summary.md                # Summary presentation text
└── README.md                           # Project Documentation
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
* Place it inside the `data/PlantVillage/` folder

### Run Jupyter Notebooks

All notebooks are located in the `notebooks/` directory. Launch Jupyter from the project root and navigate to `notebooks/` in the Jupyter interface:

```bash
# Launch Jupyter Notebook
jupyter notebook
```

- **Week 1 – Data Processing**: Run `notebooks/week1_data_processing.ipynb`
- **Week 2 – Model Training**: Run `notebooks/week2_model_training.ipynb`
- **Week 3 – Final Model**: Run `notebooks/final_crop_disease_detection.ipynb` or `notebooks/week3_final.ipynb`

---

## 🔄 Updated Model Workflow (Week 3)  
1. **Data Loading** → Load processed dataset (Week 1 output: `data/processed_data.pkl`)  
2. **Data Augmentation** → ImageDataGenerator for increased dataset diversity  
3. **Model Loading** → Load Week 2 baseline model for continued training  
4. **Advanced Training** → Extended epochs with callbacks for optimal performance  
5. **Enhanced Evaluation** → Comprehensive metrics and visualizations  
6. **Final Outputs** →  
   - Improved model → `models/crop_disease_model_week3.h5`  
   - Enhanced history → `results/training_history_week3.pkl`  
   - Final notebook → `notebooks/week3_final.ipynb`

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
1. **`final_crop_disease_detection.ipynb`** (< 10 MB) - Complete project notebook with all 3 weeks
2. **PowerPoint Presentation** (8-15 slides, < 10 MB) - Project summary and results
3. **`Week3_PPT_Summary.md`** - Detailed presentation content and analysis

### For GitHub Repository:
- All project files including models, data, and notebooks
- Complete documentation and progress tracking
- Performance analysis and technical insights

### Project Deliverables Completed:
- ✅ Data processing and EDA (Week 1)
- ✅ Baseline CNN model training (Week 2) 
- ✅ Enhanced model with augmentation (Week 3)
- ✅ Comprehensive performance analysis
- ✅ Technical documentation and insights
- ✅ Student-friendly code with explanations

---

## 👤 Author

**Name:** ARUN KUMAR L  
**Program:** AICTE Cycle 3 (2025) – Sustainable Agriculture Internship  
**Theme:** Crop Disease Detection using AI/ML  
**GitHub:** https://github.com/ARUN-L-KUMAR/AICTE-Sustainable-Agriculture  

✨ *Final Submission: 14 Sept 2025 (Week 3 Complete)*
