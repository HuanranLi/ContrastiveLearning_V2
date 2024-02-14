from data_CIFAR10 import *
from model_SimCLR import *
import argparse
import torch
import torchvision
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

def main(args):
    mlf_logger = MLFlowLogger(experiment_name=args.experiment_name, run_name = f'run_{args.run_index}', save_dir = '../logs')
    mlf_logger.log_hyperparams(args)

    if args.train_transform == 'SimCLR':
        train_transform = SimCLRTransform(input_size=args.input_size)
    else:
        raise ValueError(f'Dataset {args.train_transform} not implemented.')


    if args.dataset == 'CIFAR10':
        CIFAR10_train = torchvision.datasets.CIFAR10(
            root='../datasets', train=True, download=True
        )
        CIFAR10_test = torchvision.datasets.CIFAR10(
            root='../datasets', train=False, download=True
        )
        data_module = CIFAR10DataModule(args.input_size,
                                        args.batch_size,
                                        args.num_workers,
                                        train_transform,
                                        CIFAR10_train,
                                        CIFAR10_test)
    else:
        raise ValueError(f'Dataset {args.dataset} not implemented.')

    if args.model == 'SimCLR':
        model = SimCLRModel(max_epochs=args.max_epochs)
    else:
        raise ValueError(f'Model {args.model} not implemented.')

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                            devices=args.devices,
                            accelerator=args.accelerator,
                            logger=mlf_logger)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for CIFAR10 using PyTorch Lightning")

    parser.add_argument('--experiment_name', type=str, default='experiment_name', help='Name of the experiment.')
    parser.add_argument('--run_index', type=int, default=0, help='Index of the run within the experiment.')
    parser.add_argument('--input_size', type=int, default=128, help='Input size of the images')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices for training')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--model', type=str, default='SimCLR',help='Model to use')
    parser.add_argument('--train_transform', type=str, default='SimCLR',help='Model to use')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator for PyTorch Lightning Trainer')

    args = parser.parse_args()

    main(args)
