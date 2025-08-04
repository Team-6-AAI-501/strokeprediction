# aai-501-stroke-prediction
Group 6 Team Project

# Stroke Prediction Using Machine Learning

This project aims to develop and evaluate machine learning models for predicting stroke occurrence based on clinical and demographic data. The goal is to assist healthcare professionals in early identification of individuals at high risk of stroke using explainable AI approaches.

---
## Folder Structure
```
.
├── LICENSE
├── README.md
├── requirements.txt
├── dataset
│   └── strokeDataSet.csv
└── jupyternotebook
    └── aai501StrokePrediction.ipynb
```
## Project Structure

- `strokeDataSet.csv`: Original dataset containing features such as age, BMI, hypertension, heart disease, and smoking status.
- `aai501StrokePrediction.ipynb`: Jupyter notebook with all code steps from data preprocessing to model evaluation.

---

## Problem Statement

Stroke is one of the leading causes of death and long-term disability worldwide. Early detection can significantly improve patient outcomes. This project uses AI to predict the likelihood of stroke using various classifiers trained on patient data.

---

## Methodology

### 1. **Data Preprocessing**

- Dropped or imputed missing values (e.g., BMI)
- Label encoding and one-hot encoding of categorical variables (e.g., gender, smoking status)
- Feature scaling using `StandardScaler`

### 2. **Exploratory Data Analysis (EDA)**

- Stroke incidence visualized across demographic groups
- Correlation matrix for features
- Class imbalance identified

### 3. **Class Imbalance Handling**

- **SMOTE (Synthetic Minority Over-sampling Technique)** used to balance the dataset

### 4. **Models Trained**

- **Logistic Regression**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **Artificial Neural Network (ANN)**
- **Support Vector Machine (SVM)**

### 5. **Evaluation Metrics**

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve & AUC Score


## Key Features Used

- Age
- Hypertension
- Heart Disease
- Average Glucose Level
- BMI
- Smoking Status
- Work Type
- Marital Status

---

## Requirements

```bash
python>=3.8
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
imblearn
tensorflow
keras
shap
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repo:

```bash
git clone https://github.com/Team-6-AAI-501/strokeprediction
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook aai501StrokePrediction.ipynb
```

3. Follow the notebook cells to preprocess data, train models, and evaluate results.

## Contributors
- Pros Loung
   - email: ploung@sandiego.edu
   - git: https://github.com/ploung1234
- Quang (Andrew) Tran
   - email: andrewtran6789@gmail.com, qtran@sandiego.edu
   - git: https://github.com/Andrew2ndSun
- Surya Prakash
   - email:
   - git: https://github.com/prakashsurya-840

## License
This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments
- USD AAI-501 Course (Professor Andrew Van Benschoten)
- Dataset Contributors (https://ieee-dataport.org/documents/stroke-prediction-dataset)
- Team Members (listed above) for their ongoing collaboration and code reviews.
