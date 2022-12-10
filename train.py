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

practice_spectrogram, practice_speaker_id = train_data_contrast[0]
#print(type(practice_spectrogram))

#SKIP
# Visualize some examples
#When working with MFCC's, print out the MFCC tensors ig
pl.seed_everything(42)
NUM_IMAGES = 4
imgs = torch.stack([img for idx in range(NUM_IMAGES) for img in train_data_contrast[idx][0]], dim=0)
img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(10,5))
plt.title('Augmented image examples of the Flickr dataset')
plt.imshow(img_grid)
plt.axis('off')
plt.show()
plt.close()

def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_models'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=False, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    #trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    # if os.path.isfile(pretrained_filename):
    #     print(f'Found pretrained model at {pretrained_filename}, loading...')
    #     model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    # else:
    train_loader = data.DataLoader(train_data_contrast, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(val_data_contrast, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    pl.seed_everything(42) # To be reproducable
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model
  
  #do grid search on different hyperparameters here, only final are represented
  simclr_model = train_simclr(batch_size=256,
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=500)
