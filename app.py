import joblib
import streamlit as st
import numpy as np

# Load the pre-trained models and scalers
heart_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\heart_model.pkl")
heart_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\heart_scaler.pkl")

diabetes_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\diabetes_model.pkl")
diabetes_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\scaler_diabetes.pkl")

stroke_model = joblib.load(r"C:\Users\devar\Downloads\heart_project\stroke_model_optimized.pkl")
stroke_scaler = joblib.load(r"C:\Users\devar\Downloads\heart_project\scaler_stroke.pkl")

st.set_page_config(page_title="Health Prediction App", layout="centered")
st.title("🩺 Unified Health Risk Prediction App")

tab1, tab2, tab3 = st.tabs(["❤️ Heart Disease", "🩸 Diabetes", "🧠 Stroke"])

# -------------------- HEART DISEASE --------------------
with tab1:
    st.header("Heart Disease Prediction")

    Age = st.number_input("Age", min_value=0, key="heart_age")
    Sex = st.selectbox("Sex", ["Male", "Female"], key="heart_sex")
    RestingBP = st.number_input("RestingBP", key="heart_bp")
    Cholesterol = st.number_input("Cholesterol", key="heart_chol")
    FastingBS = st.selectbox("FastingBS (1 if FBS > 120)", [0, 1], key="heart_fbs")
    MaxHR = st.number_input("MaxHR", key="heart_hr")
    ExerciseAngina = st.selectbox("ExerciseAngina", ["Yes", "No"], key="heart_angina")
    Oldpeak = st.number_input("Oldpeak", key="heart_oldpeak")
    ST_Slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"], key="heart_st_slope")
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"], key="heart_cp")
    RestingECG = st.selectbox("RestingECG", ["Normal", "ST"], key="heart_ecg")
    HR_Reserve = st.number_input("HR_Reserve", key="heart_reserve")

    if st.button("Predict Heart Disease"):
        Sex_val = 1 if Sex == "Male" else 0
        ExerciseAngina_val = 1 if ExerciseAngina == "Yes" else 0
        ChestPainType_ATA = 1 if ChestPainType == "ATA" else 0
        ChestPainType_NAP = 1 if ChestPainType == "NAP" else 0
        ChestPainType_TA = 1 if ChestPainType == "TA" else 0
        RestingECG_Normal = 1 if RestingECG == "Normal" else 0
        RestingECG_ST = 1 if RestingECG == "ST" else 0
        ST_Slope_val = 1 if ST_Slope == "Up" else (0 if ST_Slope == "Flat" else -1)

        heart_input = np.array([[Age, Sex_val, RestingBP, Cholesterol, FastingBS,
                                 MaxHR, ExerciseAngina_val, Oldpeak, ST_Slope_val,
                                 ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
                                 RestingECG_Normal, RestingECG_ST, HR_Reserve]])

        heart_input_scaled = heart_scaler.transform(heart_input)
        heart_pred = heart_model.predict_proba(heart_input_scaled)[0][1]
        st.success(f"Heart Disease Probability: {heart_pred:.2%}")

# -------------------- DIABETES --------------------
with tab2:
    st.header("Diabetes Prediction")

    Pregnancies = st.number_input("Pregnancies", min_value=0, key="diabetes_preg")
    Glucose = st.number_input("Glucose", key="diabetes_glucose")
    BloodPressure = st.number_input("BloodPressure", key="diabetes_bp")
    SkinThickness = st.number_input("SkinThickness", key="diabetes_skin")
    Insulin = st.number_input("Insulin", key="diabetes_insulin")
    BMI = st.number_input("BMI", key="diabetes_bmi")
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", key="diabetes_dpf")
    Age_d = st.number_input("Age", key="diabetes_age")

    if st.button("Predict Diabetes"):
        diabetes_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DiabetesPedigreeFunction, Age_d]])

        diabetes_input_scaled = diabetes_scaler.transform(diabetes_input)
        diabetes_pred = diabetes_model.predict_proba(diabetes_input_scaled)[0][1]
        st.success(f"Diabetes Probability: {diabetes_pred:.2%}")

# -------------------- STROKE --------------------
with tab3:
    st.header("Stroke Prediction")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="stroke_gender")
    age_s = st.number_input("Age", min_value=0.0, key="stroke_age")
    hypertension = st.selectbox("Hypertension", [0, 1], key="stroke_hypertension")
    heart_disease = st.selectbox("Heart Disease", [0, 1], key="stroke_heart_disease")
    ever_married = st.selectbox("Ever Married", ["Yes", "No"], key="stroke_married")
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], key="stroke_work")
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key="stroke_residence")
    avg_glucose_level = st.number_input("Average Glucose Level", key="stroke_glucose")
    bmi_s = st.number_input("BMI", key="stroke_bmi")
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"], key="stroke_smoking")

    if st.button("Predict Stroke"):
        try:
            age_hypertension = age_s * hypertension

            gender_Male = 1 if gender == "Male" else 0
            ever_married_Yes = 1 if ever_married == "Yes" else 0
            work_type_Private = 1 if work_type == "Private" else 0
            work_type_Self_employed = 1 if work_type == "Self-employed" else 0
            work_type_children = 1 if work_type == "children" else 0
            Residence_type_Urban = 1 if Residence_type == "Urban" else 0
            smoking_status_formerly = 1 if smoking_status == "formerly smoked" else 0
            smoking_status_never = 1 if smoking_status == "never smoked" else 0
            smoking_status_smokes = 1 if smoking_status == "smokes" else 0

            bmi_category_normal = 0
            bmi_category_overweight = 0
            bmi_category_obese = 0
            if 18.5 <= bmi_s < 25:
                bmi_category_normal = 1
            elif 25 <= bmi_s < 30:
                bmi_category_overweight = 1
            elif bmi_s >= 30:
                bmi_category_obese = 1

            stroke_input = np.array([[age_s, hypertension, heart_disease, age_hypertension,
                                      gender_Male, ever_married_Yes, work_type_Private,
                                      work_type_Self_employed, work_type_children,
                                      Residence_type_Urban, smoking_status_formerly,
                                      smoking_status_never, smoking_status_smokes,
                                      bmi_category_normal, bmi_category_overweight,
                                      bmi_category_obese]])

            st.write("Input vector (before scaling):", stroke_input)

            stroke_input_scaled = stroke_scaler.transform(stroke_input)
            st.write("Scaled input:", stroke_input_scaled)

            stroke_pred = stroke_model.predict_proba(stroke_input_scaled)[0][1]
            st.success(f"🧠 Stroke Probability: {stroke_pred:.2%}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your input values.")
            st.error("Ensure all fields are filled correctly.")
            st.error("If the problem persists, please contact support.")
            st.error("Thank you for your understanding.")
            st.error("We are working to resolve this issue as soon as possible.")
            st.error("Your patience is greatly appreciated.")
            st.error("For immediate assistance, please reach out to our support team.")
            st.error("We apologize for any inconvenience caused.")
            st.error("Thank you for using our health prediction app.")  
            