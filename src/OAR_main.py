from data_CIFAR10 import *
from OAR_model_SimCLR import *
import argparse
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import os

from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    # Logging the hyperparams
    mlf_logger = MLFlowLogger(experiment_name=args.experiment_name, run_name = args.run_name, save_dir = '../logs')
    mlf_logger.log_hyperparams(args)

    # Select the Data Augmentation / Transform
    if args.train_transform == 'SimCLR':
        train_transform = SimCLRTransform(input_size=args.input_size)
    else:
        raise ValueError(f'Data Transform {args.train_transform} not implemented.')

    # Select the dataset
    if args.dataset == 'CIFAR10':
        CIFAR10_train = torchvision.datasets.CIFAR10(
            root='../datasets', train=True, download=True
        )
        CIFAR10_test = torchvision.datasets.CIFAR10(
            root='../datasets', train=False, download=True
        )

        num_classes = 10
        data_module = CIFAR10DataModule(args.input_size,
                                        args.batch_size,
                                        args.num_workers,
                                        train_transform,
                                        CIFAR10_train,
                                        CIFAR10_test,
                                        OAR_ratio = args.OAR_ratio)

        feature_bank_size = data_module.get_feature_bank_size(args.batch_size)
        OAR_feature_bank_size = data_module.get_OAR_feature_bank_size(args.batch_size)

        mlf_logger.log_hyperparams({'feature_bank_size': feature_bank_size})
        print(f'feature_bank_size:', feature_bank_size)
        print(f'OAR_feature_bank_size:', OAR_feature_bank_size)

    else:
        raise ValueError(f'Dataset {args.dataset} not implemented.')

    # Select the model
    if args.model == 'SimCLR':
        model = OARSimCLRModel(max_epochs=args.max_epochs,
                            batch_size=args.batch_size,
                            feature_dim = args.feature_size,
                            feature_bank_size = feature_bank_size,
                            num_classes = num_classes,
                            temperature = args.LossTemperature,
                            learning_rate = args.learning_rate,
                            optimizer = args.optimizer,
                            OAR_ratio = args.OAR_ratio,
                            OAR_feature_bank_size = OAR_feature_bank_size)
    else:
        raise ValueError(f'Model {args.model} not implemented.')


    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    # Training
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                            devices=args.devices,
                            accelerator=args.accelerator,
                            logger=mlf_logger,
                            callbacks=[checkpoint_callback])
                            #use_distributed_sampler = False
    trainer.fit(model, data_module)
    trainer.test(ckpt_path='best',datamodule = data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for CIFAR10 using PyTorch Lightning")

    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the experiment.')
    parser.add_argument('--run_name', type=str, default='SSL', help='Name of the Run.')
    parser.add_argument('--run_index', type=int, default=0, help='Index of the run within the experiment.')

    parser.add_argument('--OAR_ratio', type=float, default=0, help='The ratio of training data for Supervised Learning')

    parser.add_argument('--model', type=str, default='SimCLR',help='Model to use')
    parser.add_argument('--num_TF_layers', type=int, default=0, help='Number of TransFusion Layers added, Default 0.')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer of the contrastive model. Default SGD.')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--input_size', type=int, default=128, help='Input size of the images')
    parser.add_argument('--feature_size', type=int, default=128, help='Output size of the encoder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--train_transform', type=str, default='SimCLR',help='Model to use')

    parser.add_argument('--learning_rate', type=float, default=6e-2, help='Learning Rate')
    parser.add_argument('--LossTemperature', type=float, default=1,
                        help='The LossTemperature parameter adjusts the influence of a particular loss function component in the overall loss computation. It is a floating-point value. Defaults to 1.')

    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices for training')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator for PyTorch Lightning Trainer')



    args = parser.parse_args()
    main(args)
