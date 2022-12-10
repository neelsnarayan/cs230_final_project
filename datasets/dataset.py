## Standard libraries
import os
from copy import deepcopy
import pandas as pd
import random

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms
import torchaudio 
from torchaudio import transforms as audio_transforms

# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    !pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import tensorboard
%load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../cs230_project/train_audio/flickr_audio/wavs"
# DATASET_PATH = "/content/drive/My Drive/cs230_final_project/train_audio"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

#make sure data is to float 16 otherwise can't use it
class FlickrDataset(Dataset):
    """Audio dataset of 40000 samples of 183 speakers."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with speaker lables.
            root_dir (string): Directory with all the .wav files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_samples = pd.read_csv(csv_file, sep=' ', header=None, names=["Audio File", "Speaker ID"])
        self.audio_samples['Speaker ID'] = self.audio_samples['Speaker ID'] - 1 #convert to zero indexing
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.audio_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                self.audio_samples.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(audio_name)
        speaker_id = self.audio_samples.iloc[idx, 1:]['Speaker ID']

        if self.transform:
            #print("waveform before transform: ", type(waveform))
            waveform = self.transform(waveform, sample_rate)
            #print("waveform after transform: ", type(waveform))
            #print(waveform)
            waveform = [torch.cat((waveform[i], waveform[i], waveform[i]), dim=0) for i in range(len(waveform))]
            #waveform = torch.cat((waveform[0], waveform[0], waveform[0]), dim=0)

        return waveform, speaker_id #transforms.ToTensor(speaker_id)
