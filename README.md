🌱 Crop Disease Detection - Week 1 (Data Processing)

This project is part of a machine learning pipeline to detect crop diseases using the PlantVillage dataset.
Week 1 focuses on data preprocessing and visualization, preparing the dataset for training deep learning models in later weeks.


---

📂 Project Structure

├── PlantVillage/              # Dataset (not included in repo, download separately)
├── week1_data_processing.py   # Python script for preprocessing & visualization
└── README.md                  # Project documentation


---

📊 Features

Loads the PlantVillage dataset (multiple crop disease categories).

Resizes all images to 128x128 pixels.

Normalizes image values (0–1 scale).

Splits dataset into training (80%) and testing (20%) sets with stratification.

Visualizes:

Class distribution across categories.

Sample images from the training set.




---

🚀 How to Run

1. Clone the Repository

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2. Download Dataset

Download the PlantVillage dataset from Kaggle:
🔗 PlantVillage Dataset

Extract it into the project folder:


<project-folder>/PlantVillage/

3. Install Requirements

pip install numpy pandas matplotlib seaborn opencv-python scikit-learn

4. Run Script

python week1_data_processing.py


---

📈 Outputs

Dataset Info: Shapes of training & test sets.

Class Distribution Plot: Bar chart of sample counts per class.

Sample Images: 9 random training images with labels.



---

🔮 Next Steps

Build a deep learning model (CNN) to classify crop diseases.

Train and evaluate performance on the processed dataset.

Deploy as a web/mobile application for farmers.



---

⚠️ Note

The dataset is not included in this repository due to size limitations.
Download it manually from Kaggle and place it in the PlantVillage/ directory.


---

👨‍💻 Author

ARUN KUMAR L
