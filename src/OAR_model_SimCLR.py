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
from Loss_SymNSQ import *
from model_TransFusion import *


class OARLoss(nn.Module):
    def __init__(self):
        super(OARLoss, self).__init__()
        # print('OARLoss Init')

    def forward(self, z_tensor, anchors):
        # Normalize the input tensor
        normalized_zs = F.normalize(z_tensor)

        z_tensor_reshaped = normalized_zs.unsqueeze(2).float()
        anchors_reshaped = anchors.unsqueeze(1).float()
        inner_product = torch.bmm(anchors_reshaped, z_tensor_reshaped)
        # print(inner_product.shape)

        return -1 * inner_product.sum()


class OARSimCLRModel(pl.LightningModule):
    def __init__(self,
                max_epochs,
                batch_size,
                feature_dim,
                feature_bank_size,
                num_classes,
                temperature,
                learning_rate,
                optimizer,
                OAR_ratio,
                OAR_feature_bank_size):

        super().__init__()
        self.save_hyperparameters()

        # Parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.feature_bank_size = feature_bank_size
        self.num_classes = num_classes
        self.knn_t = 1
        self.k_choice = [25, 50, 100, 200]
        self.lr = learning_rate
        self.optimizer = optimizer

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
        self.criterion = NTXentLoss(temperature=temperature)

        self.OAR_ratio = OAR_ratio
        if self.OAR_ratio > 0:
            torch.manual_seed(42)
            random_matrix = torch.randn(self.feature_dim, self.feature_dim)
            _, _, v = torch.svd(random_matrix)
            self.sup_anchors = v[:self.num_classes, :]

            self.OAR_criterion = OARLoss()


            self.OAR_feature_bank_size = OAR_feature_bank_size
            self.register_buffer("OAR_feature_bank", torch.randn(self.OAR_feature_bank_size, self.feature_dim))
            self.register_buffer("OAR_feature_labels", torch.randint(0, self.num_classes, (self.OAR_feature_bank_size,)))
            self.OAR_feature_bank_ptr = 0

            print('OAR mode is on! Ratio:', self.OAR_ratio)




    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
            )
        elif self.optimizer == 'Adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=5e-4
            )
        else:
            raise ValueError('Optimizer {self.optimizer} not implemented.')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

    def on_fit_start(self):
        if self.OAR_ratio > 0:
            self.sup_anchors = self.sup_anchors.to(self.device)


    def training_step(self, batch, batch_idx):
        if self.OAR_ratio == 0:
            # Forward Pass
            (x0, x1), labels, _ = SSL_batch
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion(z0, z1)

            self.log("train_loss", loss, batch_size=self.batch_size)
            self._update_feature_bank(z0, labels)

            return loss

        else:
            # print(batch)
            SSL_batch = batch[0]['SSL']
            # Forward Pass
            (x0, x1), labels, _ = SSL_batch
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion(z0, z1)

            self.log("train_loss", loss, batch_size=self.batch_size)
            self._update_feature_bank(z0, labels)


            OAR_batch = batch[0]['OAR']
            (x0, x1), labels, _ = OAR_batch

            z0 = self.forward(x0)
            z1 = self.forward(x1)
            # print(labels)
            target_anchors = self.sup_anchors[labels]

            OAR_loss = self.OAR_criterion(z0, target_anchors)
            OAR_loss += self.OAR_criterion(z1, target_anchors)
            self.log("train_OAR_loss", OAR_loss, batch_size=self.batch_size)

            OAR_loss = OAR_loss / 2 / len(labels)

            return loss + OAR_loss

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z


    # def _update_feature_bank(self, features, labels):
    #     batch_size = features.size(0)
    #     ptr = self.feature_bank_ptr
    #     assert self.feature_bank_size % batch_size == 0  # for simplicity
    #
    #     # Replace the features at ptr (oldest features first)
    #     self.feature_bank[ptr:ptr + batch_size, :] = features.detach()
    #     self.feature_labels[ptr:ptr + batch_size] = labels.detach()
    #
    #     # Move the pointer
    #     self.feature_bank_ptr = (self.feature_bank_ptr + batch_size) % self.feature_bank_size


    def _update_feature_bank(self, features, labels, OAR=False):
        if OAR:
            feature_bank = self.OAR_feature_bank
            feature_labels = self.OAR_feature_labels
            feature_bank_ptr = self.OAR_feature_bank_ptr
            feature_bank_size = self.OAR_feature_bank_size
        else:
            feature_bank = self.feature_bank
            feature_labels = self.feature_labels
            feature_bank_ptr = self.feature_bank_ptr
            feature_bank_size = self.feature_bank_size

        batch_size = features.size(0)
        ptr = feature_bank_ptr
        assert feature_bank_size % batch_size == 0  # for simplicity

        feature_bank[ptr:ptr + batch_size, :] = features.detach()
        feature_labels[ptr:ptr + batch_size] = labels.detach()
        if OAR:
            self.OAR_feature_bank_ptr = (ptr + batch_size) % feature_bank_size
        else:
            self.feature_bank_ptr = (ptr + batch_size) % feature_bank_size


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
            total = batch_labels.size(0)
            accuracy = correct / total
            self.log(f"eval_{k}-NN_accuracy", accuracy, batch_size=self.batch_size)

        return loss



    def on_test_start(self):
        if self.OAR_ratio > 0:
            OAR_training_loader = self.trainer.datamodule.OAR_vanilla_training_loader()
            for batch in OAR_training_loader:
                images, labels, _ = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                z = self.forward(images)
                self._update_feature_bank(z, labels, OAR=True)

        # Assuming vanilla_training_loader is a DataLoader in the datamodule
        vanilla_training_loader = self.trainer.datamodule.vanilla_training_loader()

        for batch in vanilla_training_loader:

            images, labels, _ = batch

            # Move images and labels to the device
            images = images.to(self.device)
            labels = labels.to(self.device)

            z = self.forward(images)

            # Update feature bank with embeddings and labels
            self._update_feature_bank(z, labels)


    def test_step(self, batch, batch_idx):
        images, labels, _ = batch

        z = self.forward(images)

        for k in self.k_choice:
            # # Run kNN prediction for z using the feature bank
            # Seed for reproducibility

            # Original test with 100% of the feature bank
            pred_labels = knn_predict(z, self.feature_bank, self.feature_labels, self.num_classes, k, self.knn_t)
            correct = (pred_labels == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            self.log(f"test_{k}-NN_accuracy_100", accuracy, batch_size=self.batch_size)

            if self.OAR_ratio > 0:
                pred_labels = knn_predict(z, self.OAR_feature_bank, self.OAR_feature_labels, self.num_classes, k, self.knn_t)
                metric_name = f"test_{k}-NN_accuracy_OAR{int(self.OAR_ratio * 100)}"
            else:
                sampled_feature_bank, sampled_feature_labels = sample_feature_bank(self.feature_bank, self.feature_labels, 0.10)
                pred_labels = knn_predict(z, sampled_feature_bank, sampled_feature_labels, self.num_classes, k, self.knn_t)
                metric_name = f"test_{k}-NN_accuracy_10"

            correct = (pred_labels == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            self.log(metric_name, accuracy, batch_size=self.batch_size)

        return {"test_accuracy": accuracy}



# Function to sample a percentage of the feature bank
def sample_feature_bank(feature_bank, feature_labels, percentage):
    random_seed = 123
    torch.manual_seed(random_seed)
    num_samples = int(len(feature_bank) * percentage)
    indices = torch.randperm(len(feature_bank))[:num_samples]
    return feature_bank[indices], feature_labels[indices]
