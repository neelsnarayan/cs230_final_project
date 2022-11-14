import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim


# Generate a bare bones embedding of the audio
class MLP(torch.nn.Module):
    def __init__(self, embedding_size, hidden_sizes, output_size):
        super().__init__()
        model_layers = []
        layer_inp_size = embedding_size
        for h in hidden_sizes:
            model_layers.append(torch.nn.Linear(layer_inp_size, h))
            model_layers.append(torch.nn.ReLU())
            layer_inp_size = h
        model_layers.append(torch.nn.Linear(layer_inp_size, output_size))
        self.model = torch.nn.Sequential(*model_layers)
        # self.model = torch.nn.Sequential(
        # 	torch.nn.Linear(embedding_size, hidden_size),
        # 	torch.nn.ReLU(),
        # 	torch.nn.Linear(hidden_size, output_size),
        # )
        print(
            f"MLP Model Parameters: Input size {embedding_size}, Hidden size {hidden_sizes}, Output size {output_size}"
        )

    def forward(self, x):
        return self.model(x)


# TODO change these filepaths
TRAIN_PATH_DIR_IN = "train_audio/"
TRAIN_PATH_DIR_OUT = "train_features/"
TRAIN_PATH_DIR_OUT_LABELS = TRAIN_PATH_DIR_OUT + "labels.pkl"
TRAIN_PATH_LABELS = "wav2spk.txt"


def import_data():
    df = pd.read_csv(TRAIN_PATH_LABELS, sep=' ', header=None, names=["Audio File", "Speaker ID"])
    df["Audio Features"] = df["Audio File"].apply(lambda filename: filename.split("/")[-1].replace(".wav", ".npy"))

    inputs = []
    labels = []
    file_to_speaker_id = dict(zip(df['Audio Features'], df['Speaker ID']))
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


def train_mlp_model():
    # torch_seed = 69420
    sklearn_seed = 420
    model_name = "mlp"

    print("\n************************************* IMPORTING DATA *************************************\n")
    inputs, labels = import_data()
    print("we have this many inputs: ", inputs.shape)
    print("we have this many labels: ", labels.shape)


    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=sklearn_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=sklearn_seed)

    X_train, X_val, X_test = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test)
    X_train, X_val, X_test = torch.tensor(X_train, dtype=torch.float32), \
                torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_val, y_test = torch.tensor(y_train, dtype=torch.long), \
                torch.tensor(y_val, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    output_size = labels.shape[0]
    embedding_size = inputs.shape[1]

    print(f"Input shape: {inputs.shape} Label shape: {labels.shape} Num Labels: {output_size}")

    # X_train, X_val, X_test = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test)
    print("\n************************************* FINISHED PROCESSING DATA *************************************\n")
    hidden_sizess = [[512, 128]]
    for hidden_sizes in hidden_sizess:
        print(
            f"\n************************************* BEGINNING TRAINING {hidden_sizes} *************************************\n")
        lr = 0.001
        model = MLP(embedding_size=embedding_size, hidden_sizes=hidden_sizes, output_size=output_size)
        print(model)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        val_patience = 50
        val_loss_inc = 0
        val_min_dif = 0
        run_half_epochs = True

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        n_epoch = 100

        print(
            f"Training Parameters: Num Epochs {n_epoch}, Lr {lr}, Patience {val_patience}, Min Dif {val_min_dif}, Run Half Epochs: {run_half_epochs}")
        for i in range(n_epoch):
            optimizer.zero_grad()

            train_preds = model(X_train)
            train_loss = loss(train_preds, y_train)
            train_losses.append(train_loss.item())
            train_loss.backward()

            optimizer.step()

            train_acc = torch.mean((torch.argmax(train_preds, dim=1) == y_train), dtype=torch.float32).item()
            train_accs.append(train_acc)

            val_preds = model(X_val)
            val_loss = loss(val_preds, y_val)
            val_losses.append(val_loss.item())
            val_acc = torch.mean((torch.argmax(val_preds, dim=1) == y_val), dtype=torch.float32).item()
            val_accs.append(val_acc)

            if i % 10 == 0:
                print(
                    f"Epoch {i}: Train Loss: {train_losses[-1]} Train Acc: {train_accs[-1]} Val Loss: {val_losses[-1]} Val Acc: {val_accs[-1]}"
                )
            if (len(val_losses) > 1):
                val_loss_inc = val_loss_inc + 1 if (val_losses[-2] + val_min_dif < val_losses[-1]) else 0
            if val_loss_inc > val_patience and (i > n_epoch / 2 or not run_half_epochs):
                print(f"Breaking at Epoch {i} because val loss increasing: {val_losses[-5:]}")
                break

        test_preds = model(X_test)
        test_acc = torch.mean((torch.argmax(test_preds, dim=1) == y_test), dtype=torch.float32).item()
        print("Test Acc: ", test_acc, " Test Acc w/ Best Model: ", test_acc)
        print("\n************************************* FINISHED TRAINING *************************************\n")

train_mlp_model()
