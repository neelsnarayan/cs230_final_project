"""
simclr_algorithm.py implements the SimCLR algorithm, which is a framework of contrastive
learning for visual representations. To harness the visual power of SimCLR, we rely on the
waveform visualizations of each audio file and then apply contrastive learning. Here, we
train a model and then generate metrics to gauge effectiveness.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            num_classes=4 * hidden_dim
        )  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        # we use Adam optimization with weight decay
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        """
        Calculate noise-contrastive estimation training loss and log results for visualization
        with tensor board
        """
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],  # First position positive example
                cos_sim.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def train_simclr(
        self,
        device_type,
        batch_size,
        checkpoint_path,
        train_constrast,
        val_contrast,
        num_workers,
        max_epochs=500,
        **kwargs,
    ):
        # generate trainer to train with simclr
        trainer = pl.Trainer(
            default_root_dir=os.path.join(checkpoint_path, "saved_models"),
            accelerator="gpu" if str(device_type).startswith("cuda") else "cpu",
            devices=1,
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=False, mode="max", monitor="val_acc_top5"
                ),
                LearningRateMonitor("epoch"),
            ],
        )

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(checkpoint_path, "SimCLR.ckpt")
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            model = SimCLR.load_from_checkpoint(
                pretrained_filename
            )  # Automatically loads the model with the saved hyperparameters
        else:
            train_loader = data.DataLoader(
                train_constrast,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=num_workers,
            )
            val_loader = data.DataLoader(
                val_contrast,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=num_workers,
            )
            pl.seed_everything(42)  # To be reproducable
            model = SimCLR(max_epochs=max_epochs, **kwargs)
            trainer.fit(model, train_loader, val_loader)
            model = SimCLR.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )  # Load best checkpoint after training

        return model

    def training_step(self, batch, batch_idx):
        """
        Calculate NCE loss on training set
        """
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        """
        Calculate NCE loss on validation set
        """
        self.info_nce_loss(batch, mode="val")
