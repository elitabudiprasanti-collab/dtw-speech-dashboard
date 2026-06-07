
import streamlit as st
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

from dtw_model import hitung_dtw

folder_path = "dataset/AudioWAV"

df = pd.read_csv("sample_data.csv")

st.title("Dashboard DTW Speech Emotion")

st.sidebar.header("Pilih Audio")

audio1 = st.sidebar.selectbox(
    "Audio Referensi",
    df["filename"]
)

audio2 = st.sidebar.selectbox(
    "Audio Pembanding",
    df["filename"]
)

path1 = os.path.join(
    folder_path,
    audio1
)

y, sr = librosa.load(
    path1,
    sr=None
)

st.subheader("Waveform")

fig, ax = plt.subplots(
    figsize=(10,3)
)

librosa.display.waveshow(
    y,
    sr=sr,
    ax=ax
)

st.pyplot(fig)

st.subheader("MFCC")

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

fig, ax = plt.subplots(
    figsize=(10,4)
)

img = librosa.display.specshow(
    mfcc,
    x_axis="time",
    ax=ax
)

plt.colorbar(img)

st.pyplot(fig)

distance = hitung_dtw(
    audio1,
    audio2
)

st.subheader("DTW Distance")

st.metric(
    label="Distance",
    value=round(distance,2)
)

emotion = df[
    df["filename"] == audio2
]["emotion"].values[0]

st.subheader("Prediksi Emosi")

st.success(
    f"Emosi audio pembanding : {emotion}"
)
