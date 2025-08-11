import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_recommend_model.joblib")

# Streamlit page config
st.set_page_config(page_title="🎓 Placement Recommendation", layout="centered")

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Placement Recommendation System")
st.markdown("Fill in the student details to check if they are **recommended for campus placement**.")

# ===== FORM INPUT =====
with st.form("placement_form"):
    student_name = st.text_input("Student Name")
    
    col1, col2 = st.columns(2)
    with col1:
        overall_grade = st.selectbox("Overall Grade", ["A", "B", "C", "D", "E"])
        research_score = st.number_input("Research Score", min_value=0, max_value=100, step=1)
    with col2:
        obedient = st.selectbox("Obedient", ["Yes", "No"])
        project_score = st.number_input("Project Score", min_value=0, max_value=100, step=1)

    submit_btn = st.form_submit_button("🔍 Predict Recommendation")

# ===== PREDICTION =====
if submit_btn:
    input_df = pd.DataFrame([[overall_grade, obedient, research_score, project_score]],
                            columns=["OverallGrade", "Obedient", "ResearchScore", "ProjectScore"])
    
    prediction = model.predict(input_df)[0]

    st.subheader(f"📌 Prediction for {student_name if student_name else 'the student'}:")
    if prediction == "Yes":
        st.success("✅ Recommended for campus placement!")
    else:
        st.error("❌ Not recommended for campus placement.")
