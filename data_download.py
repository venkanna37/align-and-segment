"""
Download and extract the data
List of available datasets:
- train: train and validation sets
- test: test set
- trained_models: all trained models
- raw_data: The raw data of train and validation sets
- dinov2_features: The DINOv2 features extracted from from all buildings: train, validation, and test sets
"""

import os
import argparse
from torchvision.datasets.utils import download_url, extract_archive


# List of available datasets
datasets = ['khartoum.zip', 'lasvegas.zip', 'paris', 'sanjuan.zip']


def download_data(dataset_name: str, outdir: str):
    """ Download the data """
    # check filename and define the URL
    filename = f"{dataset_name}.zip"
    if dataset_name in datasets:
        url = f"https://sid.erda.dk/share_redirect/fOxSHwH5hr/{filename}"
    else:
        raise ValueError(f"Dataset/ Model name: {dataset_name} is not available.")

    # Download and extract file.
    download_url(url, ".", filename)
    print(f"Downloaded {filename}")
    extract_archive(filename, outdir)
    print(f"Extracted {filename} to {outdir}")
    os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="filename to download", default="lumi")
    parser.add_argument("--outdir", type=str, help="Out directory to extract the zip file", default="./temp")
    args = parser.parse_args()
    download_data(args.filename, args.outdir)
