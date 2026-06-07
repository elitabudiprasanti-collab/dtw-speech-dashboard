import os
import librosa

from dtw import dtw

# lokasi folder audio
folder_path = "dataset/AudioWAV"

def hitung_dtw(file1, file2):

    path1 = os.path.join(
        folder_path,
        file1
    )

    path2 = os.path.join(
        folder_path,
        file2
    )

    y1, sr1 = librosa.load(
        path1,
        sr=None
    )

    y2, sr2 = librosa.load(
        path2,
        sr=None
    )

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