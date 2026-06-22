"""
# List of models for Building Alignment

"""

import argparse
from tools import training

# model, training and data details, can be modified here or using command line arguments
params = {
    'wb_project_name': 'Align'
}

cities = {
    'Khartoum': 'AOI_5_Khartoum_Train',
    'Shanghai': 'AOI_4_Shanghai_Train',
    'Paris': 'AOI_3_Paris_Train',
    'Vegas': 'AOI_2_Vegas_Train',
    'San': 'AOI_9_San_Juan'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("--keyword", type=str, default="test")
    parser.add_argument("--server", type=str, help="Server", default="local")

    # data parameters
    parser.add_argument("--city", type=str, help="One of four cities", default='Vegas')
    # cities list: Vegas, Khartoum, Shanghai, Paris
    parser.add_argument("--hold_city", type=str, help="One of four cities", default=None)
    parser.add_argument("--noise_type", type=str, help="Noise method either u or b", default='u')
    # not it is noise level in pixels
    parser.add_argument("--synth_method", type=int, help="Synthetic data method", default=50)
    parser.add_argument("--aug_shift", type=int, help="Noise in shift", default=10)
    parser.add_argument("--sample_size", type=int, help="sample_size", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=2)
    parser.add_argument("--patch_size", type=int, help="Training epochs", default=320)
    parser.add_argument("--single_index", type=int, help="Single index in DF for testing", default=None)
    parser.add_argument("--use_snet_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_tnet_aug", action=argparse.BooleanOptionalAction, default=True)

    # model parameters
    parser.add_argument("--model_name", type=str, help="UNet Spatial transformer", default="method4")
    parser.add_argument("--tnet_backbone", type=str, help="TNet backbone name", default="vitsmall")
    parser.add_argument("--snet_backbone", type=str, help="SNet backbone name", default=None)
    parser.add_argument("--use_tnet_weights", action=argparse.BooleanOptionalAction, default=False)

    # training parameters
    parser.add_argument("--epochs", type=int, help="Training epochs", default=300)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--lr_drop", type=int, help="Learning drop", default=15)  #not using now
    parser.add_argument("--use_reg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg_loss_type", type=str, help="loss_type", default='mse')
    parser.add_argument("--seg_loss_type", type=str, help="loss_type", default='cross_entropy')
    parser.add_argument("--reg_loss_wt", type=float, help="Weight for the affine loss", default=100)
    parser.add_argument("--do_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre_weights", type=str, help="pretrained_weights", default=None)

    # visualisation parameters
    parser.add_argument("--use_wb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write_images", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    params['keyword'] = args.keyword

    # data parameters
    params['city'] = cities[args.city]
    params['hold_city'] = args.hold_city
    params['synth_method'] = args.synth_method
    params['aug_shift'] = args.aug_shift
    params['sample_size'] = args.sample_size
    params['batch_size'] = args.batch_size
    params['patch_size'] = args.patch_size
    params['single_index'] = args.single_index
    params['use_snet_aug'] = args.use_snet_aug
    params['use_tnet_aug'] = args.use_tnet_aug
    params['noise_type'] = args.noise_type

    # model parameters
    params['model_name'] = args.model_name
    params['tnet_backbone'] = args.tnet_backbone
    params['snet_backbone'] = args.snet_backbone
    params['use_tnet_weights'] = args.use_tnet_weights

    # training parameters
    params['epochs'] = args.epochs
    params['lr_drop'] = args.lr_drop
    params['learning_rate'] = args.learning_rate
    params['reg_loss_wt'] = args.reg_loss_wt
    params['use_reg'] = args.use_reg
    params['do_val'] = args.do_val
    params['reg_loss_type'] = args.reg_loss_type
    params['seg_loss_type'] = args.seg_loss_type
    params['pre_weights'] = args.pre_weights

    # visualisation parameters
    params['use_wb'] = args.use_wb
    params['write_images'] = args.write_images

    # set data and output directories based on server
    if args.server == "local":
        params['data_dir'] = "./sample_data"
        params['checkpoints_dir'] = "./temp"
    # elif args.server == "another_machine":
    #     # add directories if you are running on any server
    #     pass
    else:
        raise ValueError("Directories not defined")

    train = training.AlignTraining(**params)

    train.train()


