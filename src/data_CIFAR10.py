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

from lightning.pytorch.utilities import CombinedLoader

# CIFAR10 Data Module
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, input_size, batch_size, num_workers, train_transform, torch_train_dataset, torch_test_dataset, val_split=0.2, OAR_ratio = 0.0):
        super().__init__()
        # parameter
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.OAR_ratio = OAR_ratio

        # Data Transform
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

        # Pytorch Dataset
        self.cifar10_train_torch = torch_train_dataset
        self.cifar10_test_torch = torch_test_dataset

    def get_feature_bank_size(self, batch_size):
        # Train-Val Split
        train_size = int((1 - self.val_split) * len(self.cifar10_train_torch))

        return train_size // batch_size * batch_size


    def get_OAR_feature_bank_size(self, batch_size):
        # Train-Val Split
        train_size = int(self.OAR_ratio * len(self.cifar10_train_torch))

        return train_size // batch_size * batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_dataset = data.LightlyDataset.from_torch_dataset(self.cifar10_train_torch, transform = self.train_transform)

            # Train-Val Split
            train_size = int((1 - self.val_split) * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.cifar10_train, self.cifar10_val = random_split(full_train_dataset, [train_size, val_size])

            if self.OAR_ratio > 0:
                OAR_size = int(self.OAR_ratio * len(full_train_dataset))
                left_size = len(full_train_dataset) - OAR_size
                torch.manual_seed(42)
                self.OAR_train, _ = random_split(full_train_dataset, [OAR_size, left_size])

        if stage == 'test' or stage is None:
            self.cifar10_test = data.LightlyDataset.from_torch_dataset(self.cifar10_test_torch, transform = self.test_transform)
            self.vanilla_train_loader = data.LightlyDataset.from_torch_dataset(self.cifar10_train_torch, transform = self.test_transform)

            if self.OAR_ratio > 0:
                OAR_vanilla_train_loader = data.LightlyDataset.from_torch_dataset(self.cifar10_train_torch, transform = self.test_transform)
                OAR_size = int(self.OAR_ratio * len(OAR_vanilla_train_loader))
                left_size = len(OAR_vanilla_train_loader) - OAR_size
                torch.manual_seed(42)
                self.OAR_vanilla_trainset, _ = random_split(OAR_vanilla_train_loader, [OAR_size, left_size])



    def train_dataloader(self):
        SSL_loader = DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

        if self.OAR_ratio == 0:
            return SSL_loader
        else:
            OAR_loader = self.OAR_training_loader()
            combined_loader = CombinedLoader({'SSL': SSL_loader, 'OAR': OAR_loader}, mode="max_size_cycle")
            print('Combined Loader!')
            return combined_loader

    def vanilla_training_loader(self):
        return DataLoader(
            self.vanilla_train_loader,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def OAR_vanilla_training_loader(self):
        return DataLoader(
            self.OAR_vanilla_trainset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def OAR_training_loader(self):
        return DataLoader(
            self.OAR_train,
            batch_size=self.batch_size,
            shuffle=False,
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
