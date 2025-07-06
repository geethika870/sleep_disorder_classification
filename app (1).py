import streamlit as st
import pandas as pd
import numpy as np
import joblib # or pickle, depending on how you saved your model

# --- Load the trained model and preprocessor ---
# Replace 'your_best_model.pkl' and 'your_preprocessor.pkl' with the actual file paths
# You would typically save your trained model and the fitted preprocessor (scaler, encoder)
# after the training and optimization steps.
try:
    # Example: Loading a scikit-learn model (like the hybrid-optimized models)
    # If your best model is a Keras model (CNN-LSTM), you would use tensorflow.keras.models.load_model
    best_model = joblib.load('best_hybrid_model.pkl') # Assuming this is the saved hybrid model
    # Assuming 'preprocessor.pkl' contains the fitted ColumnTransformer or scaler/encoder pipeline
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or preprocessor file not found. Please make sure 'best_hybrid_model.pkl' and 'preprocessor.pkl' are in the correct path.")
    best_model = None
    preprocessor = None
except Exception as e:
    st.error(f"An error occurred while loading the model or preprocessor: {e}")
    best_model = None
    preprocessor = None


# --- Streamlit App Title and Description ---
st.title("Sleep Disorder Prediction")
st.write("Enter the patient's information to predict the sleep disorder.")

# --- User Input Section ---
st.header("Patient Information")

# Example input fields - replace with the actual features from your dataset
# You'll need to know the data types and ranges of your features
# Ensure the order of these inputs matches the order of features expected by your preprocessor
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
occupation = st.text_input("Occupation") # Consider using a selectbox for known categories
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
physical_activity_level = st.number_input("Physical Activity Level (minutes per week)", min_value=0, value=50)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Normal Weight"]) # Adjust categories
blood_pressure_systolic = st.number_input("Blood Pressure (Systolic)", min_value=50, value=120)
blood_pressure_diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=30, value=80)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, value=8000)


# --- Prediction Button ---
if st.button("Predict Sleep Disorder"):
    if best_model is not None and preprocessor is not None:
        # --- Prepare input data for the model ---
        # Create a dictionary from user inputs
        input_data = {
            'Gender': gender,
            'Age': age,
            'Occupation': occupation,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity_level,
            'Stress Level': stress_level,
            'BMI Category': bmi_category,
            # Combine systolic and diastolic BP if your model expects a single 'Blood Pressure' feature like in the original data
            'Blood Pressure': f"{blood_pressure_systolic}/{blood_pressure_diastolic}",
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply the same preprocessing as used during training
        # Ensure the column order and names match the training data
        try:
            # The preprocessor should handle the 'Blood Pressure' string format and other transformations
            # If your preprocessor expects specific column order, you might need to reorder input_df columns
            # based on X_train.columns before transforming. Assuming preprocessor handles this.
            input_processed = preprocessor.transform(input_df)

            # --- Make Prediction ---
            # If your best model is CNN-LSTM, you'll need to reshape the input_processed
            # For a standard scikit-learn model, prediction is direct
            prediction = best_model.predict(input_processed)


            # --- Display Prediction Result ---
            # You'll need to map the numerical prediction back to the original class label
            # This requires the label_encoder used during preprocessing
            # Assuming you saved the label_encoder or know the mapping:
            # predicted_disorder = label_encoder.inverse_transform(prediction) # If you saved the encoder

            # For now, just display the numerical prediction
            st.subheader("Prediction:")
            # Assuming the target variable 'Sleep Disorder' was encoded as 0, 1, 2
            # You need to know the mapping from your LabelEncoder fit
            # Example mapping (replace with your actual mapping based on your LabelEncoder fit)
            disorder_mapping = {0: 'No Disorder', 1: 'Insomnia', 2: 'Sleep Apnea'} # Replace with your mapping
            predicted_disorder_name = disorder_mapping.get(prediction[0], "Unknown Disorder")
            st.write(f"Predicted Sleep Disorder: {predicted_disorder_name}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Model or preprocessor not loaded. Please check the file paths.")

# --- How to run this script ---
# 1. Save your trained model and fitted preprocessor (StandardScaler, OneHotEncoder, LabelEncoder)
#    using joblib or pickle after the training and optimization steps.
#    Make sure the file paths in the script match where you saved them.
# 2. Save this code as a Python file (e.g., app.py). (Done by the cell above)
# 3. Open your terminal or command prompt, navigate to the directory where you saved the file.
# 4. Run the command: streamlit run app.py (Use the command below in a new code cell)
