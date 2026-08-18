# Align and Segment (AnS): Unsupervised Learning for Building Segmentation From Misaligned Labels

[![arXiv](https://img.shields.io/badge/arXiv-2607.10841-b31b1b.svg)](https://arxiv.org/pdf/2607.10841)
[![Paper](https://img.shields.io/badge/-Paper-orange?logo=huggingface&logoColor=FFD21E&labelColor=555)](https://huggingface.co/papers/2607.10841)
[![Dataset](https://img.shields.io/badge/-Dataset-orange?logo=huggingface&logoColor=FFD21E&labelColor=555)](https://huggingface.co/datasets/venkanna37/align-and-segment)
[![Weights](https://img.shields.io/badge/-Weights-orange?logo=huggingface&logoColor=FFD21E&labelColor=555)](https://huggingface.co/venkanna37/align-and-segment)

This is the official repository for the ECCV 2026 paper **"Align and Segment: Unsupervised Learning for Building Segmentation From Misaligned Labels."**

This paper proposes a method for aligning and segmenting buildings from misaligned labels **without using any golden labels**.

The outline of this repository is given as follows:

* [**Requirements**](#requirements)
* [**Datasets and Weights**](#datasets-and-weights)
* [**Train**](#train)
* [**Test**](#test)
* [**Cite**](#cite)

---

## Requirements

Installing the packages in `requirements.txt` allows the pretrained weights and test results to be reproduced.

```bash
conda create -n "ans" python=3.11.0
pip install -r requirements.txt
```

---

## Datasets and Weights

Two of the derived datasets and pretrained weights are hosted on Hugging Face and are
downloaded automatically during training and testing.
The separate [data generator](https://github.com/venkanna37/align-and-segment/blob/main/tools/datagen/rebo_datagen.py)
also prepared for [ReBO](https://huggingface.co/datasets/kevinlikai/ReBO) dataset,
it automatically download the data from Hugging Face and prepare for training.
All these datasets and weights can also be downloaded separately from links below:

- [AnS Datasets](https://huggingface.co/datasets/venkanna37/align-and-segment)
- [ReBO Dataset](https://huggingface.co/datasets/kevinlikai/ReBO)
- [Weights](https://huggingface.co/venkanna37/align-and-segment)

To avoid redistributing DINOv3 weights and code, the [`timm`](https://timm.fast.ai/) library was used in this implementation,
which automatically downloads only weights of DINOv3 encoder during training and testing.
The weights of remaining parts of complete method, SNet and TNet are released through Hugging Face.
The results can be reproduced by combining weights of DINOv3, SNet and TNet.

Custom datasets should follow the folder structure below.
The `train.py` and `test.py` scripts automatically download and arrange
both weights and datasets into this same folder structure.

### Folder structure

```
datasets               # All datasets
└── lasvegas           # Las Vegas: Synthetic data
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
└── rebo               # Checkpoints on Las Vegas data with systematic noise
    └── ...            # Same structure as lasvegas_u
```

---

## Train

To train the model on `lasvegas` with random noise, run:

```bash
python train.py --keyword lasvegas_u --dataset_name lasvegas --noise_type u
```

Change `--dataset_name` to `sanjuan` or `rebo` to reproduce the training
weights on the respective dataset.

Use `--noise_type` to `u` or `b` when training on the `lasvegas` dataset for generating
random and systematic noises, respectively. It is not required for both `sanjuan` and `rebo` datasets.

For all available options, run:

```bash
python train.py --help
```

---

## Test

To evaluate the model on the test set of Las Vegas with random noise:

```bash
python test.py --keyword lasvegas_u --dataset_name lasvegas --noise_type u
```

The folder names inside `runs/`, or on the [Hugging Face Model page](https://huggingface.co/venkanna37/align-and-segment), correspond to the `--keyword` values used during training.

Adjust `--keyword` and `--dataset_name` accordingly to use pretrained weights and datasets, respectively.

---

## Note
All commands that can be used for training and testing added to `commands.txt` file.
This repository's implementation uses DINOv3 weights loaded via the timm Python library.
The original paper's approach instead requires submitting a [request form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) to download the DINOv3 weights directly from Meta.
Because the layer names differ between the timm-loaded weights and the weights obtained through the request form,
we retrained these models and are sharing the resulting weights for SNet and TNet.
Because of retraining, there may be very small deviation of results compared results reported in original paper.

---

## Cite
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