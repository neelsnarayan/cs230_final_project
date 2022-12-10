"""
main.py trains on the SimCLR algorithm.
"""

import os
import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms


import contrastive_transformations
import dataloading
import simclr_algorithm

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "/content/drive/My Drive/cs230_final_project/train_audio"

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"

# We use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Sanity checking to make sure we are on GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# Step 1: Generate audio transforms
contrast_transforms = torchvision.transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=128),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Step 2: Load datasets with contrastive transformations applied
train_csv_path = "/content/drive/My Drive/cs230_final_project/wav2spk_train.txt"
train_data_contrast = dataloading.FlickrDataset(
    csv_file=train_csv_path,
    root_dir=DATASET_PATH,
    transform=contrastive_transformations.ContrastiveTransformations(
        contrast_transforms
    ),
)
val_csv_path = "/content/drive/My Drive/cs230_final_project/wav2spk_val.txt"
val_data_contrast = dataloading.FlickrDataset(
    csv_file=val_csv_path,
    root_dir=DATASET_PATH,
    transform=contrastive_transformations.ContrastiveTransformations(
        contrast_transforms
    ),
)

# Step 3: Train SimCLR model
simclr_model = simclr_algorithm.train_simclr(
    device_type=device,
    batch_size=4,
    checkpoint_path=CHECKPOINT_PATH,
    train_contrast=train_data_contrast,
    val_contrast=val_data_contrast,
    hidden_dim=128,
    lr=5e-4,
    temperature=0.07,
    weight_decay=1e-4,
    max_epochs=10,
)

# Step 4: Visualize results with tensor boards
""" This is done in a jupyter notebook using the tensor boards module """
