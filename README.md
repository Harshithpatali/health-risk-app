# ML_AI_health_app


# ❤️ Heart Disease & 🧠 Stroke Prediction Module

This module is part of the [ML_AI_health_app](https://github.com/Harshithpatali/ML_AI_health_app) project. It provides an easy-to-use interface for predicting the probability of **Heart Disease** and **Stroke** using trained machine learning models. Built with **Streamlit**, it offers real-time risk estimation based on user-provided medical inputs.

---

## 🗂️ Contents


---

## 🧠 Models Used

- `heart_model.pkl`: Likely a logistic regression or ensemble classifier trained on preprocessed heart disease data with 15 features.
- `stroke_model.pkl`: Trained model using a similar approach, with 16 input features (after one-hot encoding).

Each model is trained with feature scaling using `StandardScaler`, and categorical variables are one-hot encoded to match input shape.

---

## ⚙️ Features

### 1. Heart Disease Prediction
- Inputs: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, ECG, Max Heart Rate, Oldpeak, Exercise Angina, etc.
- Processing: Manual one-hot encoding for categorical features
- Output: Probability of having heart disease

### 2. Stroke Prediction
- Inputs: Age, Gender, Hypertension, Heart Disease, Marital Status, Work Type, Glucose, BMI, Smoking Status, etc.
- Processing: Manual one-hot encoding and scaling
- Output: Probability of having a stroke

---

## 🚀 How to Run

```bash
# From inside the heart_project folder
streamlit run app.py
