# 🩺 Diabetes Prediction Project  

This is a simple Machine Learning project where we predict whether a person is diabetic or not based on their health information. The project also provides helpful lifestyle tips (do’s and don’ts) depending on the prediction result.  

---

## 📌 Features of the Project  
- Uses **KNN Imputer** to handle missing values in `Insulin` and `SkinThickness`.  
- Compares different ML models like Logistic Regression, Decision Tree, Random Forest, XGBoost, and AdaBoost.  
- **AdaBoost** was chosen as the final model with the best performance.  
- Built an interactive **Streamlit app** where users can:  
  - Enter their health details (like Glucose, BMI, Age, etc.).  
  - Get an instant prediction (Diabetic / Not Diabetic).  
  - See a personalized checklist of **Do’s and Don’ts** for managing their health.  
- Integrated **Gemini LLM API** to generate the checklist dynamically.  

---
📂 Project Structure

│-- streamlitapi.py              # Streamlit app for deployment  
│-- cleaning.ipynb # Jupyter notebook with ML pipeline  
│-- ada_classification_model.pkl # Saved ML model  
│-- ada_scaler_model.pkl # Saved Scaler   
│-- README.md            # Project description  

📊 Dataset Used
The dataset is the PIMA Indians Diabetes Dataset, which contains medical information such as:

Glucose

Blood Pressure

BMI

Insulin

Skin Thickness

Age

Outcome (Diabetic or Not)

🙋‍♂️ Why This Project?
This project was built as a student learning project to practice:

Data cleaning and preprocessing

Model training and evaluation

Model deployment using Streamlit

API integration with LLMs (Gemini API)

🌟 Future Improvements
Improve model accuracy using advanced techniques.

Add more health features for better predictions.

Enhance UI for a better user experience.

🧑‍💻 Author
Kirti Agarwal - a student learning Machine Learning.
