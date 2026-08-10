⚠️ **This code is under preparation. It will be ready soon!**

# Align and Segment (AnS): Unsupervised Learning for Building Segmentation From Misaligned Labels

[![arXiv](https://img.shields.io/badge/arXiv-2607.10841-b31b1b.svg)](https://arxiv.org/pdf/2607.10841)
[![Papers](https://img.shields.io/badge/-Papers-orange?logo=huggingface&logoColor=FFD21E&labelColor=black)](https://huggingface.co/papers/2607.10841)
[![Dataset](https://img.shields.io/badge/-Dataset-orange?logo=huggingface&logoColor=FFD21E&labelColor=black)](https://huggingface.co/datasets/venkanna37/align-and-segment)
[![Model](https://img.shields.io/badge/-Model-orange?logo=huggingface&logoColor=FFD21E&labelColor=black)](https://huggingface.co/venkanna37/align-and-segment)

This is the official repository for the ECCV 2026 paper **"Align and Segment: Unsupervised Learning for Building Segmentation From Misaligned Labels."**

This paper proposes a method for aligning and segmenting buildings from misaligned labels **without using any golden labels**.

The outline of this repository is given as follows:

* [**Requirements**](#requirements)
* [**Datasets and Weights**](#datasets-and-weights)
* [**Train**](#train)
* [**Test**](#test)
* [**Citing**](#citing)

---

## Requirements

Installing the packages in `requirements.txt` allows the trained weights and test results to be reproduced.

```bash
conda create -n "ans" python=3.11.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

---

## Datasets and Weights

All derived datasets and checkpoints are hosted on Hugging Face and are downloaded automatically during training and evaluation. They can also be downloaded separately:

- [Datasets](https://huggingface.co/datasets/venkanna37/align-and-segment)
- [Weights](https://huggingface.co/venkanna37/align-and-segment)

To avoid redistributing DINOv3 weights and code, the [`timm`](https://timm.fast.ai/) library is used, which automatically downloads the weights during training and testing.
The decoder weights for SNet and TNet are shared so the results can be reproduced.

Custom datasets should follow the folder structure below. The `train.py` and `test.py` scripts automatically download and arrange both weights and datasets into this same structure.

### Folder structure

```
datasets               # All datasets
└── lasvegas           # Las Vegas data
    ├── train          # Training set
    │   ├── images     # Input images
    │   └── labels     # Input labels
    ├── val            # Validation set
    │   └── ...        # Same structure as train
    └── test           # Test set
        └── ...        # Same structure as train

runs
├── lasvegas_u         # Checkpoints on Las Vegas data with random noise
│   ├── decoder.pt     # Decoder weights
│   └── tnet.pt        # TNet weights
└── lasvegas_b         # Checkpoints on Las Vegas data with systematic noise
    └── ...            # Same structure as lasvegas_u
```

---

## Train

To train the model on `sample_data`, run:

```bash
python train.py --keyword test_run --dataset_name sample_data
```

Change `--dataset_name` to `paris`, `khartoum`, `lasvegas`, `sanjuan`, or `rebo` to reproduce the training weights on the respective dataset.

Use `--noise_type` to select **random** (`u`) or **systematic** (`b`) noise when training on the `paris`, `khartoum`, or `lasvegas` datasets.

For all available options, run:

```bash
python train.py --help
```

---

## Test

To evaluate the model on the test set of Las Vegas with random noise:

```bash
python test.py --keyword lasvegas_u --dataset_name lasvegas
```

The folder names inside `runs/`, or on the [Hugging Face Model page](https://huggingface.co/venkanna37/align-and-segment), correspond to the `--keyword` values used during training.

Change `--dataset_name` to `paris`, `khartoum`, `lasvegas`, `sanjuan`, or `rebo` to reproduce the results on respective dataset.

---

## Citing
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{Guthula_align2026,
  title={Align and Segment: Unsupervised Learning for Building Segmentation From Misaligned Labels},
  author={Venkanna Babu Guthula and Oswin Krause and Dimitri Gominski and Hui Zhang and Johan Mottelson and Ankit Kariryaa and Nico Lang and Christian Igel},
  booktitle=European Conference on Computer Vision (ECCV),
  month = {September},
  year={2026}
}
```