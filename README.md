# Lifestyle & Health Risk Prediction - Machine Learning Project

A complete machine learning project for predicting health risk levels (high/low) based on lifestyle factors using various classification algorithms.

## 📊 Dataset

**Source**: [Kaggle - Lifestyle and Health Risk Prediction Dataset](https://www.kaggle.com/datasets/zahranusrat/lifestyle-and-health-risk-prediction-dataset?resource=download)

**Description**:  dataset containing lifestyle and health-related variables for 5000 individuals.

**Features**:
- **Numerical**: age, weight, height, sleep (hours), BMI
- **Categorical**: exercise level, sugar intake, smoking, alcohol consumption, married status, profession

**Target Variable**: `health_risk` (binary: high/low)

## 📁 Project Structure

```
ml project/
│
├── README.md                                          # This file
├── Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset (1).csv  # Original dataset
│
├── Projet_Lifestyle_HealthRisk_EDA.ipynb            # Exploratory Data Analysis
├── Projet_Lifestyle_HealthRisk_Preprocessing.ipynb   # Data preprocessing
├── Projet_Lifestyle_HealthRisk_Modeling.ipynb        # Model training and evaluation
│
└── preprocessed_data/                                # Saved preprocessed data
    ├── X_train_balanced.npy
    ├── y_train_balanced.npy
    ├── X_test_scaled.npy
    ├── y_test.npy
    ├── feature_names.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── label_encoders_binary.pkl
    └── preprocessing_info.pkl
```

## 🔬 Notebooks Overview

### 1. EDA (Exploratory Data Analysis)
**File**: `Projet_Lifestyle_HealthRisk_EDA.ipynb`

**Contents**:
- Dataset loading and basic information
- Missing values analysis
- Target variable distribution
- Numerical variables distributions (histograms, boxplots)
- Categorical variables distributions
- Correlation analysis
- Relationships between features and target variable

**Goal**: Understand the data structure, distributions, and relationships before modeling.

### 2. Preprocessing
**File**: `Projet_Lifestyle_HealthRisk_Preprocessing.ipynb`

**Contents**:
1. Data loading
2. Feature/target separation
3. **Optimized encoding**:
   - Binary variables (2 categories): Label Encoding (0/1) - one column
   - Multi-category variables (>2): One-Hot Encoding
4. Target variable encoding
5. Train/Test split (80/20, stratified)
6. Standardization of numerical variables
7. **Class imbalance handling**: Conservative SMOTE (60/40 balance)
8. Saving preprocessed data and preprocessing objects


### 3. Modeling
**File**: `Projet_Lifestyle_HealthRisk_Modeling.ipynb`

**Contents**:
1. Loading preprocessed data
2. Utility functions for consistent evaluation
3. **6 Models tested**:
   - **Dummy Classifier** (baseline)
   - **Logistic Regression**
   - **Polynomial Features + Logistic Regression** (degrees 1, 2, 3)
   - **Decision Tree** (various max_depth values)
   - **k-Nearest Neighbors** (various k values)
   - **Random Forest** (various hyperparameter combinations)
4. Model comparison and selection
5. ROC curves comparison
6. Conclusion and interpretation

**Evaluation Metrics**:
- Cross-validation (5-fold StratifiedKFold)
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion matrices
- Train vs Test comparison (overfitting detection)

## 🎯 Results Summary

### Best Model: Decision Tree (max_depth=10)

**Performance**:
- **CV Accuracy**: 0.9924 (±0.0013)
- **Test Accuracy**: 0.9970
- **Test F1-Score**: 0.9950
- **Test ROC-AUC**: 0.9969

**Model Ranking** (by Test Accuracy):
1. Decision Tree: 0.9970
2. Random Forest: 0.9930
3. Polynomial Logistic Regression: 0.9340
4. Logistic Regression: 0.8890
5. kNN: 0.8870
6. Dummy Classifier: 0.6980

### Key Findings

- **Overfitting observed**: Polynomial Logistic Regression (train: 0.9904, test: 0.9340)
- **Best generalization**: Decision Tree shows minimal overfitting (train: 0.9998, test: 0.9970)
- **Most important features** (Random Forest): Age, BMI, Weight, lifestyle factors

## 🛠️ Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn (for SMOTE)
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 🚀 Usage

### Step 1: Run EDA
Execute `Projet_Lifestyle_HealthRisk_EDA.ipynb` to explore the dataset.

### Step 2: Run Preprocessing
Execute `Projet_Lifestyle_HealthRisk_Preprocessing.ipynb` to:
- Encode categorical variables
- Standardize numerical features
- Handle class imbalance with SMOTE
- Save preprocessed data

### Step 3: Run Modeling
Execute `Projet_Lifestyle_HealthRisk_Modeling.ipynb` to:
- Load preprocessed data
- Train and evaluate all models
- Compare performances
- Select best model


## 📈 Model Interpretability

Based on Random Forest feature importance:
- **Age** (highest importance)
- **BMI** (second highest)
- **Weight**
- Lifestyle factors (exercise, sleep, sugar intake)
- Behavioral factors (smoking, alcohol)

This aligns with medical knowledge: age and BMI are strong predictors of health risk.



Machine Learning Project - Albert School
