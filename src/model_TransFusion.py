
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from util_affinity_matrix import *
from lightly.models.modules.heads import ProjectionHead, SimCLRProjectionHead



class AttentionBlock(nn.Module):
    def __init__(self, input_size, output_size, activation = 'relu'):
        super(AttentionBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        #self.query = nn.Sequential(nn.Linear(input_size, output_size), nn.LayerNorm(output_size))
        #self.key = nn.Sequential(nn.Linear(input_size, output_size), nn.LayerNorm(output_size))
        #self.value = nn.Sequential(nn.Linear(input_size, output_size), nn.LayerNorm(output_size))

        self.query = nn.Linear(input_size, output_size)
        self.key = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)

        if activation == 'relu':
            print(f'Activation: ReLU')
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            print(f'Activation: Sigmoid')
            self.activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            print(f'Activation: LeakyRelu')
            self.activation = nn.LeakyReLU(0.1)
        else:
            assert False


    def forward(self, x):
        q =  F.normalize(self.query(x))
        k =  F.normalize(self.key(x))
        v =  self.value(x)

        cos_sim = torch.matmul(q, k.transpose(-2, -1))
        #attn_weights = nn.functional.relu(attn_weights)
        attn_weights = self.activation(cos_sim)
        attn_weights = normalize_with_diagonal_zero(attn_weights)

        # Apply the attention weights to the value vectors
        output = torch.matmul(attn_weights, v) + x

        return output

class TransFusionProjectionHead(SimCLRProjectionHead):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_TF_layers: int,
        style = 'before_FNN',
    ):
        print('TransFusion Init with number of layers: ', num_TF_layers)

        if style == 'before_FNN':
            super().__init__(input_dim, hidden_dim, output_dim)
            TF_layers = [layer for _ in range(num_TF_layers) for layer in [nn.LayerNorm(input_dim), AttentionBlock(input_dim, input_dim)]]
            self.layers = nn.Sequential(*TF_layers, *self.layers)
        elif style == 'after_FNN':
            super().__init__(input_dim, hidden_dim, output_dim)
            TF_layers = [layer for _ in range(num_TF_layers) for layer in [nn.LayerNorm(output_dim), AttentionBlock(output_dim, output_dim)]]
            self.layers = nn.Sequential(*self.layers, *TF_layers)
        elif style == 'inserted_FNN':
            super().__init__(input_dim, hidden_dim, output_dim)
            TF_layers = [layer for _ in range(num_TF_layers) for layer in [nn.LayerNorm(output_dim),
                                                                            AttentionBlock(output_dim, output_dim),
                                                                            nn.Linear(output_dim, output_dim),
                                                                            nn.ReLU()]]
            self.layers = nn.Sequential(*self.layers , *TF_layers)
        else:
            raise ValueError('style for TF not implemented: ', style)

class SimCLR_TFsize_ProjectionHead(SimCLRProjectionHead):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_TF_layers: int
    ):
        super().__init__(input_dim, hidden_dim, output_dim)
        print('Equivelent TransFusion Init with number of layers: ', num_TF_layers)

        TF_layers = [layer for _ in range(num_TF_layers*3) for layer in [nn.LayerNorm(input_dim), nn.Linear(input_size, input_size)]]

        self.layers = nn.Sequential(*TF_layers, *self.layers)
