"""
# Testing trained AnS model
"""

import os
import torch
import argparse
import pandas as pd

from pipelines.evaluate import evaluate_ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--keyword", type=str,
                        help='keyword used in saving pretrained model', default="test")
    parser.add_argument("--server", type=str, default="lumi")
    parser.add_argument("--set_name", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--wt_file", type=str, default="") #default should be 300
    parser.add_argument("--csv_file", type=str, default="aug_reg.csv")
    parser.add_argument("--sample_size", type=int, help="sample_size", default=None)
    parser.add_argument("--checkpoints_dir", type=str,
                        help='output directory to save models, logs', default="./runs")
    parser.add_argument("--data_dir", type=str,
                        help='data directory with input data', default="sample_data/vegas")

    args = parser.parse_args()
    params = vars(args)
    params['weights_path'] = os.path.join(params['checkpoints_dir'], args.keyword, f'best{args.wt_file}.pth')
    model_weights = torch.load(params['weights_path'], map_location='cpu')
    params['patch_size'] = model_weights['params']['patch_size']
    params['dataset_type'] = model_weights['params']['dataset_type']

    # set data and output directories based on server
    if params['dataset_type'] == "synthetic":
        params['noise_type'] = model_weights['params']['noise_type']
        params['synth_method'] = model_weights['params']['synth_method']

    params['model_name'] = model_weights['params']['model_name']
    params['tnet_backbone'] = model_weights['params']['tnet_backbone']

    print(f"Last epoch: {model_weights['epoch']}")
    del model_weights

    # initialize training object and train model
    train = evaluate_ans.AlignPrediction(**params)
    metrics = train.predict()
    metrics["keyword"] = args.keyword

    # create csv file if does not exist
    csv_path = args.csv_file
    df = pd.DataFrame([metrics])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print("Created new csv file and added metrics")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print("Added metrics to existing existing csv file")


