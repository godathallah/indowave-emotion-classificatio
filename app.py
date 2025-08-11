import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model dan label encoder yang sudah disimpan
model = load_model('model_bilstm_indowave.keras')
le = joblib.load('label_encoder.joblib')

def extract_mfcc_sequence(data, sr, max_pad_len=130):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T

st.title("IndoWaveSentiment Emotion Classification")

uploaded_file = st.file_uploader("Upload file audio .wav sesuai format IndoWaveSentiment", type=["wav"])

if uploaded_file is not None:
    data, sr = librosa.load(uploaded_file, sr=None)
    features = extract_mfcc_sequence(data, sr)
    features = features.reshape(1, features.shape[0], features.shape[1])

    prediction_prob = model.predict(features)[0]
    pred_class = np.argmax(prediction_prob)
    emotion_label = le.inverse_transform([pred_class])[0]
    confidence = prediction_prob[pred_class]

    st.write(f"**Hasil Prediksi Emosi:** {emotion_label}")
    st.write(f"**Confidence:** {confidence:.2f}")
