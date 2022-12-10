"""
clustering.py is used for testing our model on the unseen JL corpus dataset. We
train our model on the Flickr Dataset and then use spectral clustering built on
our LSTM model to group the speakers in the JL corpus into 4 groups, since there
are 4 speakers present.

NOTE:
This version of clustering.py works for our LSTM embeddings. However, we've slightly
modified our clustering algorithm for compliance with our SimCLR output.
"""

import keras as keras
import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity
from sklearn.cluster import SpectralClustering
import time

MODEL_NAME = "softmax_sparse_categorical_crossentropy"
TEST_PATH_LABELS = "wav2spk_test.txt"
TEST_PATH_DIR_OUT = "test_features/"


def load_test_data():
    """
    Here, we load our test data and ensure that our labels match our inputs. We return
    two arrays containing our audio files and the speaker for each audio file
    """
    df = pd.read_csv(
        TEST_PATH_LABELS, sep=" ", header=None, names=["Audio File", "Speaker ID"]
    )
    df["Audio Features"] = df["Audio File"].apply(
        lambda filename: filename.split("/")[-1].replace(".wav", ".npy")
    )

    # load input and label arrays
    inputs = []
    labels = []
    file_to_speaker_id = dict(zip(df["Audio Features"], df["Speaker ID"]))
    for j, filename in enumerate(file_to_speaker_id):
        try:
            input_file = np.load(TEST_PATH_DIR_OUT + filename)
            try:
                inputs.append(input_file)
                labels.append(file_to_speaker_id[filename] - 1)
            except:
                print("Couldn't open the label")
                raise
        except:
            print("This input isn't part of our feature set yet")
            raise

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels


if __name__ == "__main__":

    """
    START MODEL
    """
    """
    We create a sequential model with two LSTM layers (each with 64 units),
    two dense layers (one with 350 units and one with 500 units) with dropout
    present in the 350 unit dense layer (keep probability of 70%), and a final
    activation layer. ReLU and softmax activations are used.
    """
    # model = keras.Sequential()

    # model.add(keras.layers.LSTM(64, return_sequences=True))
    # model.add(keras.layers.LSTM(64))

    # model.add(keras.layers.Dense(350, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # model.add(keras.layers.Dense(500))
    # model.add(keras.layers.Activation(keras.activations.softmax))
    """
    END MODEL
    """

    """
    Since we've already trained the model, we go ahead and load it
    """
    # load the model
    model = keras.models.load_model(MODEL_NAME)
    # status checks
    print("Finished loading model")

    # pop off the last dense + softmax layer - design choice
    # model.pop()
    # print("Popped off last dense + softmax")

    # generate optimizer
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
    print("Finished predictions")

    start = time.time()
    # we use 4 clusters since there are 4 speakers in the JL Corpus Dataset (our test set)
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
            predictions[Segment(i, i + segment)] = clustering.labels_[i]

        print("Finished annotations")

        # calculate metrics
        diarizationErrorRate = DiarizationErrorRate()
        diarizationPurity = DiarizationPurity()
        DER = diarizationErrorRate(
            gt, predictions, uem=Segment(0, len(test_inputs)), detailed=True
        )
        DP = diarizationPurity.compute_components(
            gt, predictions, uem=Segment(0, len(test_inputs)), detailed=True
        )
        print(DER)
        print(DP)
