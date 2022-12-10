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

#I am not resampling bc it is all sampled at same rate
#Apply a frequency and time masking transformation to generate augmentations of the same sample, then generate MFCCs
class ContrastiveTransformations(object):
    def __init__(self, base_transforms, use_MFCC=False, n_views=2):
        self.base_transforms = base_transforms
        self.use_MFCC = use_MFCC
        self.n_views = n_views

    def __call__(self, x, sample_rate):
        num_channels, num_frames = x.shape
        create_spectrogram = torchaudio.transforms.Spectrogram(n_fft=800)
        spectrogram = create_spectrogram(x)
        #mask_spectrogram = torchaudio.transforms.TimeMasking(80)
        stretch_spectrogram = torchaudio.transforms.TimeStretch(n_freq = spectrogram.shape[1])
        #spectrogram = stretch_spectrogram(spectrogram, random.uniform(0.5, 1.5))
        spectrogram = spectrogram*3
        #print(type(spectrogram))
        #spectrogram = ToPILImage(spectrogram)
        # print(spectrogram.shape)
        # augmented_spectrograms = []
        # for i in range(self.n_views):
        #   augmented_spectrogram = stretch_spectrogram(spectrogram, random.uniform(0.5, 1.5))
        #   print(augmented_spectrogram.shape)
        #   augmented_spectrograms.append(augmented_spectrogram)
        augmented_spectrograms = [stretch_spectrogram(spectrogram, random.uniform(0.5, 1.5)) for i in range(self.n_views)]
        #smallest_n_frames = min([spectrogram.shape[3] for spectrogram in augmented_spectrograms])
        #augmented_spectrograms = [self.base_transforms(augmented_spectrogram) for augmented_spectrogram in augmented_spectrograms]
        augmented_spectrograms = [self.base_transforms(spectrogram) for i in range(self.n_views)]
        #print(type(augmented_spectrograms[0]))
        if self.use_MFCC:
          create_MFCC = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=40)
          return [create_MFCC(i) for i in augmented_spectrograms]
        else:
          return augmented_spectrograms
#TODO: maybe add noise as a transform too and get rid of some of these image transforms
# contrast_transforms = torch.nn.Sequential(
#             TimeStretch(stretch_factor, fixed_rate=True),
#             FrequencyMasking(freq_mask_param=10),
#             TimeMasking(time_mask_param=10),
#             )

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

# contrast_transforms = torchvision.transforms.Compose([transforms.ToPILImage(),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.RandomResizedCrop(size=96),
#                                           transforms.RandomApply([
#                                               transforms.ColorJitter(brightness=0.5,
#                                                                      contrast=0.5,
#                                                                      saturation=0.5,
#                                                                      hue=0.1)
#                                           ], p=0.8),
#                                           transforms.RandomGrayscale(p=0.2),
#                                           transforms.GaussianBlur(kernel_size=9),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize((0.5,), (0.5,))
#                                          ])

#TODO: make two wav2spk files --> one for train, one for val, and load these in seperately 

#unlabeled_data = FlickrDataset(root=DATASET_PATH, split='unlabeled', download=True,
#                       transform=ContrastiveTransformations(contrast_transforms, use_MFCC=False, n_views=2))
train_csv_path = "./wav2spk_TRAIN.txt"
train_data_contrast = FlickrDataset(csv_file=train_csv_path, root_dir=DATASET_PATH,
                                    transform=ContrastiveTransformations(contrast_transforms))
val_csv_path = "./wav2spk_DEV.txt"
val_data_contrast = FlickrDataset(csv_file=val_csv_path, root_dir=DATASET_PATH,
                                    transform=ContrastiveTransformations(contrast_transforms))
#unlabeled_data = FlickrDataset(root=DATASET_PATH, split='unlabeled', download=True,
#                       transform=ContrastiveTransformations(contrast_transforms, use_MFCC=False, n_views=2))


         
