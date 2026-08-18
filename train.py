"""
# Training AnS with synthetic, real and rebo dataset
"""

import argparse
from tools.training import training as training_synth, training_qualitative, training_rebo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, default='test',
                        help='keyword for saving the checkpoint and run name in weights and bias')

    # data parameters
    parser.add_argument('--dataset_name', type=str,
                        choices=['sample_data', 'lasvegas', 'sanjuan', 'rebo'],
                        default='sample_data', help='Type of dataset for training')
    parser.add_argument('--noise_type', type=str, choices=['u', 'b'], default='u',
                        help='Type of synthetic noise u: random noise, b: systematic noise')
    parser.add_argument('--misalign_magnitude', type=int,
                        choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        default=50, help='Magnitude of misalignment')
    parser.add_argument("--aug_shift", type=int, choices=[5, 10], default=10,
                        help='Noise in shift while preparing dataset')
    parser.add_argument('--max_shift', type=int, default=100,
                        help='Max shift while generating random transformation for second loss')
    parser.add_argument("--sample_size", type=int, default=None,
                        help='Select small sample of patches out of entire dataset')
    parser.add_argument("--batch_size", type=int, default=48,
                        help='Batch size of training data')
    parser.add_argument('--patch_size', type=int, default=320,
                        help='Patch size for training')
    parser.add_argument('--use_snet_aug',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Augmentation for SNet')

    # data directory parameters
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='data directory where input data available or can be downloaded')
    parser.add_argument('--checkpoints_dir', type=str,  default='./runs',
                        help='output directory to save check points and logs')

    # Model parameters
    parser.add_argument('--model_name', type=str,
                        choices=['method1', 'method2'], default='method1',
                        help="Model name is combination of types of architecture in SNet and TNet")
                        # method1: model configuration from torch.hub
                        # method2: model configuration from timm
    # check tools/models/load_models.py for more details about model_names
    parser.add_argument('--tnet_backbone', type=str, default="vitsmall",
                        help="TNet backbone name")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Learning rate')
    # Check training.py file for more details about loss_setting
    parser.add_argument('--seg_loss_type', type=str, default='cross_entropy',
                        help='First loss for SNet training')
    parser.add_argument('--reg_loss_wt', type=float, default=100,
                        help='Weight for the affine loss (lamda in paper)')
    parser.add_argument('--num_workers', type=int, default=4)

    # Visualisation parameters
    parser.add_argument('--use_wb', action=argparse.BooleanOptionalAction,
                        default=False, help='Use weights and biases for visualization or not')

    # Convert all arguments to one dictionary
    args = parser.parse_args()
    params = vars(args)

    # Initialize training object and train model
    if args.dataset_name in ['sample_data', 'lasvegas']:
        train = training_synth.AlignTraining(**params)
    elif args.dataset_name == 'sanjuan':
        params['noise_type'] = 'r'
        train = training_qualitative.AlignTraining(**params)
    elif args.dataset_name == 'rebo':
        params['patch_size'] = 512
        train = training_rebo.AlignTraining(**params)
    else:
        raise Exception("Unknown dataset type")

    train.train()


