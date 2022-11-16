import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_PATH_DIR_IN = "train_audio/wavs/"
TRAIN_PATH_DIR_OUT = "train_features/"
TRAIN_PATH_DIR_OUT_LABELS = TRAIN_PATH_DIR_OUT + "labels.pkl"
TRAIN_PATH_LABELS = "wav2spk.txt"


def load_data():
    df = pd.read_csv(
        TRAIN_PATH_LABELS, sep=" ", header=None, names=["Audio File", "Speaker ID"]
    )
    df["Audio Features"] = df["Audio File"].apply(
        lambda filename: filename.split("/")[-1].replace(".wav", ".npy")
    )

    inputs = []
    labels = []
    file_to_speaker_id = dict(zip(df["Audio Features"], df["Speaker ID"]))
    for j, filename in enumerate(file_to_speaker_id):
        try:
            input = np.load(TRAIN_PATH_DIR_OUT + filename)
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


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data()

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(350, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(500, activation="softmax"))

    return model


if __name__ == "__main__":

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2
    )

    print(X_train.shape)
    # create network
    input_shape = (X_train.shape[1], 1)  # 17, 40, 1 for initial testing
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=32,
        epochs=500,
    )

    # save model based on design features
    model.save("softmax_sparse_categorical_crossentropy")

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)
