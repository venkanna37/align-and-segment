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
                        help='keyword used in saving pretrained model', default='paris_u')
    parser.add_argument("--set_name", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--noise_type", type=str, default='u')
    parser.add_argument("--model_name", type=str, default='method1')
    parser.add_argument("--checkpoints_dir", type=str,
                        help='output directory to save models, logs', default="./runs")
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='data directory where input data available or can be downloaded')
    parser.add_argument('--dataset_name', type=str,
                        choices=['sample_data', 'lasvegas', 'sanjuan', 'rebo'],
                        default='lasvegas', help='Type of dataset for training')

    args = parser.parse_args()
    params = vars(args)

    # Initialize eval object and evaluate
    eval = evaluate_ans.AlignPrediction(**params)
    metrics = eval.evaluate()


