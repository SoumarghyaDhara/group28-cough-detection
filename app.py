import streamlit as st
import h2o
from h2o.automl import H2OAutoML
import librosa
import numpy as np

# Initialize H2O
h2o.init()

# Load the H2O dataset and train the model
@st.cache_resource
def train_h2o_model():
    try:
        # Load dataset
        data = h2o.import_file(r"C:\Users\Admin\Downloads\public_dataset_v3\coughvid_20211012\metadata_compiled.csv")

        # Split the data into training and testing sets
        train, test = data.split_frame(ratios=[0.8], seed=42)

        # Specify the features and target column
        x = data.columns[:-1]
        y = data.columns[-1]

        # Train the AutoML model
        automl = H2OAutoML(max_models=10, seed=42)
        automl.train(x=x, y=y, training_frame=train)

        return automl, test
    except Exception as e:
        st.error(f"Error training the H2O model: {e}")
        return None, None

# Extract features from an audio file
def features_extractor(file):
    # Load the audio file
    audio, sample_rate = librosa.load(file, sr=None)

    # Extract features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=16)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=10)
    mel_scaled_features = np.mean(mel_spectrogram.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_scaled_features = np.mean(zcr.T, axis=0)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_scaled_features = np.mean(spectral_centroid.T, axis=0)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    spectral_rolloff_scaled_features = np.mean(spectral_rolloff.T, axis=0)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma.T, axis=0)

    return np.hstack((mfccs_scaled_features, mel_scaled_features, zcr_scaled_features,
                      spectral_centroid_scaled_features, spectral_rolloff_scaled_features, chroma_scaled_features))

# Define the main function
def main():
    st.title("Cough Detection with H2O AutoML")
    st.write("This app uses H2O AutoML to predict whether the input corresponds to a cough sound.")

    # Train the H2O model
    automl, test = train_h2o_model()
    if automl is None:
        st.error("Failed to train the model.")
        return

    st.header("Upload an Audio File")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Extract features from the uploaded file
        features = features_extractor(audio_file)

        # Convert features into a format compatible with H2O
        features_h2o = h2o.H2OFrame([features.tolist()])

        try:
            # Predict using the H2O AutoML model
            prediction = automl.leader.predict(features_h2o)

            # Display the result
            st.write("Prediction Results:")
            st.write(prediction)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

# Shutdown H2O when the script ends
h2o.shutdown(prompt=False)
