"""
# Training AnS
"""
import sys

import argparse
from pipelines.training import training, training_qualitative, training_rebo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="test")

    # data parameters
    parser.add_argument("--data_type", type=str,
                        help="sythetic, real or rebo", default='synthetic')
    parser.add_argument("--noise_type", type=str,
                        help="u: random noise, b: systematic noise", default='u')
    parser.add_argument("--synth_method", type=int,
                        help="Magnitude of misalignment", default=50)
    parser.add_argument("--aug_shift", type=int,
                        help="Noise in shift while preparing dataset", default=10)
    parser.add_argument("--max_shift", type=int,
                        help="Max shift while generating random transformation", default=100)
    parser.add_argument("--sample_size", type=int,
                        help="Select small sample out of more number of patch", default=None)
    parser.add_argument("--batch_size", type=int,
                        help="Batch size of training data", default=48)
    parser.add_argument("--patch_size", type=int,
                        help="Patch size for training", default=320)
    parser.add_argument("--use_snet_aug",
                        help="Augmentation for SNet", action=argparse.BooleanOptionalAction, default=False)

    # data directory parameters
    parser.add_argument("--data_dir", type=str,
                        help='data directory with input data', default="sample_data/vegas")
    parser.add_argument("--checkpoints_dir", type=str,
                        help='output directory to save models, logs', default="./runs")

    # model parameters
    parser.add_argument("--model_name", type=str,
                        help="Model name is combination of architecture in SNet and TNet", default="method4c")
    # check pipelines/models/load_models.py for more details about model_names
    parser.add_argument("--tnet_backbone", type=str,
                        help="TNet backbone name", default="vitsmall")

    # training parameters
    parser.add_argument("--epochs", type=int, help="Training epochs", default=300)
    parser.add_argument("--snet_lr", type=float, help="Learning rate for SNet", default=0.00001)
    parser.add_argument("--tnet_lr", type=float, help="Learning rate for TNet", default=0.00001)
    parser.add_argument("--lr_drop", type=int, help="Learning rate drop", default=300)
    parser.add_argument("--loss_setting", type=int, help="Loss combination for TNet", default=4)
    # Check training.py file for more details about loss_setting
    parser.add_argument("--seg_loss_type", type=str, help="loss_type", default='cross_entropy')
    parser.add_argument("--reg_loss_wt", type=float,
                        help="Weight for the affine loss (lamda in paper)", default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_reg_loss", action=argparse.BooleanOptionalAction, default=False)

    # visualisation parameters
    parser.add_argument("--use_wb", action=argparse.BooleanOptionalAction, default=False)

    # convert all arguments to one dictionary
    args = parser.parse_args()
    params = vars(args)

    # initialize training object and train model
    if args.data_type == 'synthetic':
        train = training.AlignTraining(**params)
    elif args.data_type == 'real':
        params['noise_type'] = 'r'
        train = training_qualitative.AlignTraining(**params)
    elif args.data_type == 'rebo':
        train = training.AlignTraining(**params)
    else:
        raise Exception("Unknown dataset type")

    train.train()


