# Heart Disease Prediction using Machine Learning

## Overview

This project builds an end-to-end supervised machine learning pipeline to predict the presence of heart disease using patient clinical data.

The goal is to compare multiple machine learning models and identify the best-performing approach while balancing performance and interpretability.

This project was completed as part of the Machine Learning course at BITS Pilani.

---

## Problem Statement

Early detection of heart disease is critical in healthcare.

Using patient data, we aim to build a classification system that predicts whether a patient has heart disease.

The challenge is to:
- handle real-world medical data
- apply appropriate preprocessing
- compare multiple models
- evaluate performance and risks

---

## Dataset

The dataset used is the UCI Heart Disease dataset from Kaggle.

It contains features such as:
- age
- cholesterol
- blood pressure
- chest pain type
- exercise induced angina
- heart rate
- clinical indicators

The target variable was converted into binary:
- 0 = No disease
- 1 = Disease :contentReference[oaicite:1]{index=1}  

---

## Approach

### Data Preprocessing

- removed duplicate records
- handled missing values:
  - median for numeric features
  - mode for categorical features
- converted categorical variables using one-hot encoding
- applied standard scaling
- split data into 80:20 train-test using stratification :contentReference[oaicite:2]{index=2}  

---

### Baseline Models

Two baseline models were trained:

- Logistic Regression
- Naive Bayes

Results:

- Logistic Regression Accuracy = 0.8369
- Naive Bayes Accuracy = 0.8533 :contentReference[oaicite:3]{index=3}  

---

### Decision Tree

- tuned parameters:
  - max_depth = 4
  - min_samples_split = 10
  - min_samples_leaf = 5

- Accuracy = 0.8043 :contentReference[oaicite:4]{index=4}  

Key important features:
- exercise induced angina
- cholesterol
- age :contentReference[oaicite:5]{index=5}  

---

### Rule-Based Model

Rules were extracted from the decision tree.

Example rule:

If:
- exang <= threshold
- cholesterol <= threshold

Then predict Disease

Rule-based models:
- are easy to interpret
- but may reduce performance :contentReference[oaicite:6]{index=6}  

---

### kNN Model

- tested k = 1, 3, 5, 7, 9
- best performance at k = 7 and 9

Best accuracy:

- kNN Accuracy = 0.8696 :contentReference[oaicite:7]{index=7}  

Key insight:
- scaling is critical because kNN depends on distance calculations :contentReference[oaicite:8]{index=8}  

---

### Ensemble Models

Two ensemble models were used:

- Random Forest → Accuracy = 0.8587
- Gradient Boosting → Accuracy = 0.8369 :contentReference[oaicite:9]{index=9}  

---

## Final Model Comparison

| Model | Accuracy |
|------|--------|
| kNN | 0.8696 |
| Naive Bayes | 0.8533 |
| Random Forest | 0.8587 |
| Logistic Regression | 0.8369 |
| Gradient Boosting | 0.8369 |
| Decision Tree | 0.8043 |

---

## Key Insights

- kNN performed best due to similarity-based patterns in data
- Decision Tree is most interpretable
- Ensemble models improve stability but reduce explainability
- scaling significantly impacts model performance

---

## Risks in Real-World Deployment

- incorrect predictions due to poor data quality
- false negatives may miss actual disease cases
- false positives may lead to unnecessary testing
- model may not generalize across populations :contentReference[oaicite:10]{index=10}  

---

## Why This Project Matters

This project demonstrates a complete machine learning workflow:

- preprocessing
- multiple model training
- evaluation and comparison
- interpretability vs performance trade-off

It reflects real-world decision making in machine learning systems.

---

## Repository Structure
heart-disease-ml-pipeline/

├── README.md
├── src/
│ └── heart_disease_ml_pipeline.py
├── docs/
│ └── project_report.pdf
├── assets/
│ └── (charts and outputs)


---

## How to Run
Clone repository:
git clone https://github.com/chetanpant/heart-disease-ml-pipeline.git


Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub


Run:
python src/heart_disease_ml_pipeline.py


---

## Project Context
This project was developed as part of the Machine Learning course.

It covers:
- supervised learning
- classification models
- evaluation metrics
- model comparison
- interpretability
