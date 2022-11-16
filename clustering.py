import tensorflow.keras as keras
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd

MODEL_NAME = "softmax_sparse_categorical_crossentropy"
TEST_PATH_LABELS = "wav2spk_test.txt"
TEST_PATH_DIR_OUT = "test_features/"

def load_test_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    df = pd.read_csv(TEST_PATH_LABELS, sep=' ', header=None, names=["Audio File", "Speaker ID"])
    df["Audio Features"] = df["Audio File"].apply(lambda filename: filename.split("/")[-1].replace(".wav", ".npy"))

    inputs = []
    labels = []
    file_to_speaker_id = dict(zip(df['Audio Features'], df['Speaker ID']))
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

    #load the model
    model = keras.models.load_model(MODEL_NAME)

    #pop off the last dense + softmax layer (maybe change it to just pop off the softmax if possible) and recompile
    model.pop()
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #generate embeddings for all test inputs
    test_inputs, test_labels = load_test_data()

    #generate embeddings for all test inputs and cluster
    embeddings = model.predict(test_inputs, batch_size=32)
    clustering = SpectralClustering(n_clusters=4, assign_labels='discretize', random_state=0).fit(embeddings)

    gt = Annotation()
    predictions = Annotation()

    for i in range(len(test_inputs)):
        embedding = embeddings[i]
        gt[Segment(i, i+1)] = test_labels[i]
        predictions[Segment(i, i+1)] = clustering.labels_[i]

    diarizationErrorRate = DiarizationErrorRate()
    print("DER = {0:.3f}".format(diarizationErrorRate(gt, predictions, uem=Segment(0, len(test_inputs)))))


