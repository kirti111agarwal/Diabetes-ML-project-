import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

# Configure Gemini API
genai.configure(api_key="AIzaSyBo5yP3ADAsUUB6H7_ZMgSY57b39G1be7Q")
model = genai.GenerativeModel("gemini-2.0-flash")

# Load model and scaler
loaded_model = pickle.load(open("ada_classification_model.pkl", "rb"))
scaler = pickle.load(open("ada_scaler_model.pkl", "rb"))

def prediction(inputdata):
    input_data_as_numpy_array = np.array(inputdata).reshape(1, -1)
    # Log transform Insulin (5th column)
    input_data_as_numpy_array[:, 4] = np.log1p(input_data_as_numpy_array[:, 4])
    # Scale
    input_data_scaled = scaler.transform(input_data_as_numpy_array)
    # Predict
    pred = loaded_model.predict(input_data_scaled)
    return "Patient is diabetic" if pred[0] == 1 else "No diabetes detected"

def generate_checklist(diagnosis):
    if diagnosis == "Patient is diabetic":
        prompt = (
            "Write a plain text checklist for someone newly diagnosed with diabetes. "
            "Use short bullet points under the following sections: "
            "Medical, Diet, Exercise, Monitoring, Lifestyle, Self-care. "
            "Keep it concise and easy to read."
            "Just give the checklist dont use anything before and after it."
        )
    else:
        prompt = (
            "Write a plain text checklist for a non-diabetic person to maintain healthy blood sugar and prevent diabetes. "
            "Use short bullet points under the following sections: "
            "Medical, Diet, Exercise, Monitoring, Lifestyle, Self-care. "
            "Keep it concise and easy to read."
            "Just give the checklist dont use anything before and after it."
            
        )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating checklist: {e}"

def main():
    st.title("Diabetes Prediction Webapp with Checklist")

    # User inputs
    Pregnancy = st.number_input("Number of pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose level", min_value=0.0)
    BP = st.number_input("Blood Pressure", min_value=0.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0)
    Insulin = st.number_input("Insulin level", min_value=0.0)
    BMI = st.number_input("BMI", min_value=0.0)
    DiabetesPedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age of the person", min_value=0, step=1)

    if st.button("Test Result"):
        # Get diagnosis
        diagnosis = prediction([Pregnancy, Glucose, BP, SkinThickness,
                                Insulin, BMI, DiabetesPedigree, Age])
        st.success(diagnosis)

        # Generating checklist using Gemini
        with st.spinner("Generating checklist..."):
            checklist_text = generate_checklist(diagnosis)
            st.subheader("Suggested Checklist")
            st.text(checklist_text)
        

        # Example data
        df_plot = pd.DataFrame({
        "Metric": ["Glucose", "BP", "BMI", "Insulin"],
        "Patient Value": [Glucose, BP, BMI, Insulin],
        "Normal Range": [120, 80, 25, 100]
        })

        # Plot side-by-side bars
        ax = df_plot.plot(x='Metric', kind='bar', width=0.8)
        plt.ylabel("Value")
        plt.title("Patient Metrics vs Normal Range")
        plt.xticks(rotation=0)  
        st.pyplot(plt)


if __name__ == "__main__":
    main()
