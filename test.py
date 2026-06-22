"""
# List of models for Building Alignment
"""

import os
import torch
import argparse
import pandas as pd

from pipelines import evaluating

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
    parser.add_argument("--wt_file", type=str, default="best")
    parser.add_argument("--server", type=str, help="Server", default="local")
    parser.add_argument("--csv_file", type=str, help="Filename to save results", default="test.csv")

    # data parameters
    # parser.add_argument("--city", type=str, help="One of four cities", default='Vegas')
    parser.add_argument("--set_name", type=str, help="Setname to evaluate", default='test')
    parser.add_argument("--sample_size", type=int, help="sample_size", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=48)
    parser.add_argument("--patch_size", type=int, help="Training epochs", default=320)

    args = parser.parse_args()
    params['keyword'] = args.keyword

    # data parameters
    # params['city'] = cities[args.city]
    params['sample_size'] = args.sample_size
    params['batch_size'] = args.batch_size
    params['patch_size'] = args.patch_size
    params['set_name'] = args.set_name

    # set data and output directories based on server
    if args.server == "local":
        params['data_dir'] = "./sample_data"
        params['checkpoints_dir'] = "./temp"
        params['weights_path'] = os.path.join(params['checkpoints_dir'], args.keyword, f'{args.wt_file}.pth')
    # elif args.server == "another_machine":
    #     # add directories if you are running on any server
    #     pass
    else:
        raise ValueError("Server not found")

    model_weights = torch.load(params['weights_path'], map_location='cpu')
    params['synth_method'] = model_weights['params']['synth_method']
    params['model_name'] = model_weights['params']['model_name']
    params['tnet_backbone'] = model_weights['params']['tnet_backbone']
    params['city'] = model_weights['params']['city']
    params['noise_type'] = model_weights['params']['noise_type']

    print(f"Last epoch: {model_weights['epoch']}")
    del model_weights

    # initialize training object and train model
    train = evaluating.AlignPrediction(**params)
    metrics = train.predict()
    metrics["keyword"] = args.keyword
    metrics["city"] = params['city']

    # create csv file if does not exist
    csv_path = args.csv_file
    df = pd.DataFrame([metrics])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print("Created new csv file and added metrics")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print("Added metrics to existing existing csv file")


