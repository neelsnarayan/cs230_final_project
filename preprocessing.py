import os
import pickle

import librosa
import numpy as np
import pandas as pd

#TODO change these filepaths
TRAIN_PATH_DIR_IN = "train_audio/"
TRAIN_PATH_DIR_OUT = "train_features/"
TRAIN_PATH_DIR_OUT_LABELS = TRAIN_PATH_DIR_OUT + "labels.pkl"
TRAIN_PATH_LABELS = "wav2spk.txt"


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")

    #MFCCs
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features

df = pd.read_csv(TRAIN_PATH_LABELS, sep=' ', header=None, names=["Audio File", "Speaker ID"])
df["Audio Features"] = df["Audio File"].apply(
    lambda filename: filename.split("/")[-1].replace(".wav", ".npy")
)
file_to_speaker_id = dict(zip(df['Audio Features'], df['Speaker ID']))
with open(TRAIN_PATH_DIR_OUT_LABELS, "wb") as f:
    pickle.dump(file_to_speaker_id, f)

count = 0
for root, dirs, files in os.walk(TRAIN_PATH_DIR_IN):
    for file in files:
        filepath = root + os.sep + file
        if filepath.endswith(".wav"):
            features = features_extractor(filepath)
            new_filepath = TRAIN_PATH_DIR_OUT + file.replace(".wav", ".npy")
            np.save(new_filepath, features)
            count += 1
            print(new_filepath, count)
print("Total samples with features extracted: ", count)