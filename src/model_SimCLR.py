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
from util_KNN import *

class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs, batch_size, feature_dim, feature_bank_size, num_classes):
        super().__init__()

        # Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.feature_bank_size = feature_bank_size
        self.num_classes = num_classes
        self.knn_t = 1
        self.k_choice = [50, 100, 200]

        # Enable printing out sizes of each input/output
        self.example_input_array = torch.Tensor(self.batch_size, 3, 28, 28)

        # Initialize feature bank and labels
        self.register_buffer("feature_bank", torch.randn(feature_bank_size, feature_dim))
        self.register_buffer("feature_labels", torch.randint(0, num_classes, (feature_bank_size,)))
        self.feature_bank_ptr = 0

        # Backbone model
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection Head
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, feature_dim)
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        # Forward Pass
        (x0, x1), labels, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, batch_size=self.batch_size)

        # Update feature bank and labels
        batch_features = torch.cat([z0, z1], dim=0)
        batch_labels = torch.cat([labels, labels], dim=0)
        self._update_feature_bank(batch_features, batch_labels)

        return loss

    def _update_feature_bank(self, features, labels):
        batch_size = features.size(0)
        ptr = self.feature_bank_ptr
        assert self.feature_bank_size % batch_size == 0  # for simplicity

        # Replace the features at ptr (oldest features first)
        self.feature_bank[ptr:ptr + batch_size, :] = features.detach()
        self.feature_labels[ptr:ptr + batch_size] = labels.detach()

        # Move the pointer
        self.feature_bank_ptr = (self.feature_bank_ptr + batch_size) % self.feature_bank_size

    def test_step(self, batch, batch_idx):
        images, labels, _ = batch
        z = self.forward(images)

        for k in self.k_choice:
            # Run kNN prediction for z using the feature bank
            pred_labels = knn_predict(z, self.feature_bank, self.feature_labels, self.num_classes, k, self.knn_t)

            # Calculate accuracy
            correct = (pred_labels == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total

            # Log the accuracy
            self.log(f"test_{k}-NN_accuracy", accuracy, batch_size=self.batch_size)

        return {"test_accuracy": accuracy}


    def validation_step(self, batch, batch_idx):
        # Forward pass
        (x0, x1), labels, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("val_loss", loss, batch_size=self.batch_size)

        # KNN
        batch_features = torch.cat([z0, z1], dim=0)
        batch_labels = torch.cat([labels, labels], dim=0)

        for k in self.k_choice:
            pred_labels = knn_predict(batch_features, self.feature_bank, self.feature_labels, self.num_classes, k, self.knn_t)
            # Calculate accuracy
            correct = (pred_labels == batch_labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            self.log(f"eval_{k}-NN_accuracy", accuracy, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
