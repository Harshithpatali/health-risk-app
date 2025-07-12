import streamlit as st
import numpy as np
import joblib
import lightgbm as lgb
# Load all models and scalers
diabetes_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\diabetes_model.pkl")
diabetes_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\scaler_diabetes.pkl")

stroke_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\stroke_model_optimized.pkl")
stroke_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\scaler_stroke.pkl")

heart_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\heart_model.pkl")
heart_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\heart_scaler.pkl")

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

st.title("💡 Health Risk Predictor App")
st.markdown("Predict the **probability** of having **Diabetes**, **Stroke**, or **Heart Disease** based on your inputs.")

option = st.selectbox("Select Condition to Predict:", ["Diabetes", "Stroke", "Heart Disease"])

# ===============================
# 🚑 DIABETES
# ===============================
if option == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 0, 140, 70)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 100, 30)

    if st.button("Predict Diabetes Risk"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = diabetes_scaler.transform(input_data)
        prob = diabetes_model.predict_proba(scaled)[0][1]
        st.success(f"📊 Probability of Diabetes: **{prob:.2%}**")

# ===============================
# 🧠 STROKE
# ===============================
elif option == "Stroke":
    st.subheader("🧠 Stroke Prediction")
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose = st.number_input("Avg Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    if st.button("Predict Stroke Risk"):
        gender = 1 if gender == "Male" else 0
        ever_married = 1 if ever_married == "Yes" else 0
        residence = 1 if residence == "Urban" else 0

        work_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
        smoke_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

        work = work_map[work_type]
        smoke = smoke_map[smoking_status]

        raw_input = [age, hypertension, heart_disease, avg_glucose, bmi, gender, ever_married, work, residence, smoke]
        scaled = stroke_scaler.transform(np.array(raw_input).reshape(1, -1))
        prob = stroke_model.predict_proba(scaled)[0][1]
        st.success(f"📊 Probability of Stroke: **{prob:.2%}**")

# ===============================
# ❤️ HEART
# ===============================
elif option == "Stroke":
    st.subheader("🧠 Stroke Prediction")
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose = st.number_input("Avg Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    if st.button("Predict Stroke Risk"):
        # Binary features
        gender_Male = 1 if gender == "Male" else 0
        gender_Other = 1 if gender == "Other" else 0
        ever_married_Yes = 1 if ever_married == "Yes" else 0
        residence_Urban = 1 if residence == "Urban" else 0

        # One-hot for work_type (base = 'Govt_job')
        work_type_Private = 1 if work_type == 'Private' else 0
        work_type_Self = 1 if work_type == 'Self-employed' else 0
        work_type_children = 1 if work_type == 'children' else 0
        work_type_Never = 1 if work_type == 'Never_worked' else 0

        # One-hot for smoking_status (base = 'Unknown')
        smoke_former = 1 if smoking_status == 'formerly smoked' else 0
        smoke_never = 1 if smoking_status == 'never smoked' else 0
        smoke_current = 1 if smoking_status == 'smokes' else 0

        # Final input list in correct feature order
        inputs = [
            age,
            hypertension,
            heart_disease,
            avg_glucose,
            bmi,
            gender_Male,
            gender_Other,
            ever_married_Yes,
            work_type_Never,
            work_type_Private,
            work_type_Self,
            work_type_children,
            residence_Urban,
            smoke_former,
            smoke_never,
            smoke_current
        ]

        scaled = stroke_scaler.transform(np.array(inputs).reshape(1, -1))
        prob = stroke_model.predict_proba(scaled)[0][1]
        st.success(f"📊 Probability of Stroke: **{prob:.2%}**")
