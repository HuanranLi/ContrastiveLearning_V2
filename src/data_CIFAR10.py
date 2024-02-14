import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
import lightly
import lightly.data as data

# CIFAR10 Data Module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, input_size, batch_size, num_workers, train_transform, torch_train_dataset, torch_test_dataset):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            self.cifar10_train = data.LightlyDataset.from_torch_dataset(self.cifar10_train_torch, transform = self.train_transform)

        if stage == 'test' or stage is None:
            self.cifar10_test = data.LightlyDataset.from_torch_dataset(self.cifar10_test_torch, transform = self.test_transform)


    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
