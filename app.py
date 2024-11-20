import streamlit as st
import pickle
import numpy as np
import librosa
import io

# Step 1: Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("cough_detection_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Step 2: Extract features from the audio file
def features_extractor(file):
    # Load the audio file from the Streamlit uploader (it is a byte object, so we need to handle it correctly)
    audio, sample_rate = librosa.load(file, sr=None)  # sr=None keeps the original sample rate of the file

    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=16)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=10)
    mel_scaled_features = np.mean(mel_spectrogram.T, axis=0)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_scaled_features = np.mean(zcr.T, axis=0)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_scaled_features = np.mean(spectral_centroid.T, axis=0)

    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    spectral_rolloff_scaled_features = np.mean(spectral_rolloff.T, axis=0)

    # Chroma Feature
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma.T, axis=0)

    # Return all the features as a single list
    return np.hstack((mfccs_scaled_features, mel_scaled_features, zcr_scaled_features, spectral_centroid_scaled_features, spectral_rolloff_scaled_features, chroma_scaled_features))

# Step 3: Define the main function
def main():
    st.title("Cough Detection Model")
    st.write("This app predicts whether the input corresponds to a cough sound.")

    # Load the model
    model = load_model()

    # Audio file input
    st.header("Upload an Audio File")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Extract features from the audio file
        features = features_extractor(audio_file)

        # Debugging: Display feature shape
        st.write(f"Feature shape: {features.shape}")

        # Reshape the features for prediction (should be 2D array)
        features = features.reshape(1, -1)

        # Predict and display the result
        try:
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.success("Cough detected!")
            else:
                st.info("No cough detected.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if _name_ == "_main_":
    main()
