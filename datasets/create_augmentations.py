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

"""
contrastive_transformations.py applies transformations on the audio samples
for differentiation to be used with contrastive learning.
"""

import random
import torchaudio


class ContrastiveTransformations(object):
    """
    Applies transformations on audio samples and creates MFCC representations
    - Time Stretch: Randomly slow down or speed up parts of audio sample
    """

    def __init__(self, base_transforms, use_MFCC=False, n_views=2):
        self.base_transforms = base_transforms
        self.use_MFCC = use_MFCC
        self.n_views = n_views

    def __call__(self, x, sample_rate):
        num_channels, num_frames = x.shape
        create_spectrogram = torchaudio.transforms.Spectrogram(n_fft=800)
        spectrogram = create_spectrogram(x)
        # time stretch transformation
        stretch_spectrogram = torchaudio.transforms.TimeStretch(
            n_freq=spectrogram.shape[1]
        )

        spectrogram = spectrogram * 3
        augmented_spectrograms = [
            stretch_spectrogram(spectrogram, random.uniform(0.5, 1.5))
            for i in range(self.n_views)
        ]
        augmented_spectrograms = [
            self.base_transforms(spectrogram) for i in range(self.n_views)
        ]

        # generate MFCC representations for audio samples
        if self.use_MFCC:
            create_MFCC = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate, n_mfcc=40
            )
            return [create_MFCC(i) for i in augmented_spectrograms]
        else:
            return augmented_spectrograms

contrast_transforms = torchvision.transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomResizedCrop(size=128),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])

train_csv_path = "./wav2spk_TRAIN.txt"
train_data_contrast = FlickrDataset(csv_file=train_csv_path, root_dir=DATASET_PATH,
                                    transform=ContrastiveTransformations(contrast_transforms))
val_csv_path = "./wav2spk_DEV.txt"
val_data_contrast = FlickrDataset(csv_file=val_csv_path, root_dir=DATASET_PATH,
                                    transform=ContrastiveTransformations(contrast_transforms))


         
