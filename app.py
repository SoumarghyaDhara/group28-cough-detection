import streamlit as st
import pickle
import numpy as np

# Step 1: Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("cough_detection_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Step 2: Define the main function
def main():
    st.title("Cough Detection Model")
    st.write("This app predicts whether the input corresponds to a cough sound.")

    # Load the model
    model = load_model()

    # Input features
    st.header("Input Features")
    feature1 = st.number_input("Feature 1 (e.g., MFCC1):")
    feature2 = st.number_input("Feature 2 (e.g., MFCC2):")
    feature3 = st.number_input("Feature 3 (e.g., ZCR):")
    feature4 = st.number_input("Feature 4 (e.g., Spectral Centroid):")

    # Combine features into a NumPy array
    features = np.array([[feature1, feature2, feature3, feature4]])

    # Predict and display the result
    if st.button("Predict"):
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success("Cough detected!")
        else:
            st.info("No cough detected.")

if __name__ == "__main__":
    main()
