import streamlit as st

from dtw_model import predict_genre

st.title(
    "Dashboard Dynamic Time Warping (DTW)"
)

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:

    genre, distance = predict_genre(
        uploaded_file
    )

    st.success(
        f"Emosi Prediksi : {genre}"
    )

    st.write(
        f"DTW Distance : {distance:.2f}"
    )