"""
dataloading.py loads all data from the Flickr Dataset and generates waveforms
for the samples.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class FlickrDataset(Dataset):
    """
    Audio dataset of 40000 samples of 183 speakers from the Flickr Dataset.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with speaker lables.
            root_dir (string): Directory with all the .wav files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_samples = pd.read_csv(
            csv_file, sep=" ", header=None, names=["Audio File", "Speaker ID"]
        )
        self.audio_samples["Speaker ID"] = (
            self.audio_samples["Speaker ID"] - 1
        )  # convert to zero indexing
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Returns amount of audio samples
        """
        return len(self.audio_samples)

    def __getitem__(self, idx):
        """
        Returns waveform representation (graph) and speaker id for given audio
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir, self.audio_samples.iloc[idx, 0])
        # generate waveform visual representation
        waveform, sample_rate = torchaudio.load(audio_name)
        speaker_id = self.audio_samples.iloc[idx, 1:]["Speaker ID"]

        if self.transform:
            waveform = self.transform(waveform, sample_rate)
            waveform = [
                torch.cat((waveform[i], waveform[i], waveform[i]), dim=0)
                for i in range(len(waveform))
            ]

        return waveform, speaker_id  # transforms.ToTensor(speaker_id)
