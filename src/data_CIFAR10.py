import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
import lightly
import lightly.data as data

# CIFAR10 Data Module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, input_size, batch_size, num_workers, train_transform, torch_train_dataset, torch_test_dataset, val_split=0.2):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.train_transform = train_transform

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((input_size, input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )

        self.cifar10_train_torch = torch_train_dataset
        self.cifar10_test_torch = torch_test_dataset

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_dataset = data.LightlyDataset.from_torch_dataset(self.cifar10_train_torch, transform = self.train_transform)
            train_size = int((1 - self.val_split) * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.cifar10_train, self.cifar10_val = random_split(full_train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.cifar10_test = data.LightlyDataset.from_torch_dataset(self.cifar10_test_torch, transform = self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
