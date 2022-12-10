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
