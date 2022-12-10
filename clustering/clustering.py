import tensorflow.keras as keras
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
import time

MODEL_NAME = "softmax_sparse_categorical_crossentropy"
TEST_PATH_LABELS = "wav2spk_test.txt"
TEST_PATH_DIR_OUT = "test_features/"

#NOTE: For simclr version, load the simclr model instead (example in notebooks)

def load_test_data():
    df = pd.read_csv(
        TEST_PATH_LABELS, sep=" ", header=None, names=["Audio File", "Speaker ID"]
    )
    df["Audio Features"] = df["Audio File"].apply(
        lambda filename: filename.split("/")[-1].replace(".wav", ".npy")
    )

    inputs = []
    labels = []
    file_to_speaker_id = dict(zip(df["Audio Features"], df["Speaker ID"]))
    for j, filename in enumerate(file_to_speaker_id):
        try:
            input = np.load(TEST_PATH_DIR_OUT + filename)
            try:
                inputs.append(input)
                labels.append(file_to_speaker_id[filename] - 1)
            except:
                print("Couldn't open the label for some reason")
        except:
            print("This input isn't part of our feature set yet")

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels


if __name__ == "__main__":

    # model = keras.Sequential()

    # model.add(keras.layers.LSTM(64, return_sequences=True))
    # model.add(keras.layers.LSTM(64))

    # model.add(keras.layers.Dense(350, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # model.add(keras.layers.Dense(500))
    # model.add(keras.layers.Activation(keras.activations.softmax))

    # load the model
    model = keras.models.load_model(MODEL_NAME)
    # print(model.layers[-1].output)
    # print(len(model.layers))
    print("Finished loading model")

    # pop off the last dense + softmax layer (maybe change it to just pop off the softmax if possible) and recompile
    # model.pop()
    # print(model.layers[-1].output)
    # print(len(model.layers))
    # print("Popped off last dense + softmax")
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # load the test inputs
    test_inputs, test_labels = load_test_data()
    print("Loaded test inputs")

    # generate embeddings for all test inputs and cluster
    embeddings = model.predict(test_inputs, batch_size=32)
    print(test_inputs.shape)
    print(embeddings.shape)
    # labels = np.argmax(embeddings, axis=1)
    print("Finished predictions")

    start = time.time()
    clustering = SpectralClustering(
        n_clusters=4, assign_labels="discretize", random_state=0, verbose=True
    ).fit(embeddings)
    end = time.time()
    print("Clustering time: {} seconds".format(end - start))
    print("Finished clustering")

    gt = Annotation()
    predictions = Annotation()

    segment_sizes = [3, 7, 20, 50, 100, 200, 250, 350, 450, 500]

    for segment in segment_sizes:
        print("segment size: ", segment)
        for i in range(len(test_inputs)):
            embedding = embeddings[i]
            gt[Segment(i, i + segment)] = test_labels[i]
            predictions[Segment(i, i + segment)] = clustering.labels_[
                i
            ]  # labels[i]  # clustering.labels_[i]

        print("Finished annotations")

        diarizationErrorRate = DiarizationErrorRate()
        diarizationPurity = DiarizationPurity()
        DER = diarizationErrorRate(
            gt, predictions, uem=Segment(0, len(test_inputs)), detailed=True
        )
        DP = diarizationPurity.compute_components(
            gt, predictions, uem=Segment(0, len(test_inputs)), detailed=True
        )
        # print("DER = {0:.3f}".format(DER))
        print(DER)
        print(DP)
