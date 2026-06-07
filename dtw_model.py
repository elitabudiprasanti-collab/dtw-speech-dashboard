import os
import librosa
from dtw import dtw

DATASET_FOLDER = "dataset/AudioWAV"

def hitung_dtw(path1, path2):

    y1, sr1 = librosa.load(path1, sr=None)
    y2, sr2 = librosa.load(path2, sr=None)

    mfcc1 = librosa.feature.mfcc(
        y=y1,
        sr=sr1,
        n_mfcc=13
    )

    mfcc2 = librosa.feature.mfcc(
        y=y2,
        sr=sr2,
        n_mfcc=13
    )

    alignment = dtw(
        mfcc1.T,
        mfcc2.T,
        keep_internals=False
    )

    return alignment.distance


def predict_genre(uploaded_file):

    temp_file = "temp.wav"

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    hasil = []

    for file in os.listdir(DATASET_FOLDER):

        if file.endswith(".wav"):

            train_path = os.path.join(
                DATASET_FOLDER,
                file
            )

            distance = hitung_dtw(
                temp_file,
                train_path
            )

            emotion = file.split("_")[2]

            hasil.append(
                (
                    emotion,
                    distance
                )
            )

    hasil = sorted(
        hasil,
        key=lambda x: x[1]
    )

    os.remove(temp_file)

    return hasil[0]