# This code is under preparation. It will be ready soon!

# Align and Segment (AnS): Unsupervised Learning for Building Segmentation From Misaligned Labels

This is the official repository for the ECCV 2026 paper "Align and Segment:
Unsupervised Learning for Building Segmentation From Misaligned Labels".
This paper proposed a method for aligning and segmenting buildings from misaligned labels
without using any **golden labels**. The outline of this repository is given as follows.

* [**Requirements**](#requirements)
* [**Datasets**](#datasets)
* [**Train**](#train)
* [**Test**](#train-on-your-own)
* [**Align and segment**](#align-and-segment)
* [**Citing**](#citing)

## Requirements
To run our code, all packages listed in `requirements.txt` must be installed.
All experiments in the paper used a DINOv3 encoder in the segmentation network (SNet).
To reproduce the results or use the same encoder,
the [dinov3 GitHub repository](https://github.com/facebookresearch/dinov3) must
also be cloned in same directory where this code placed.
In addition, the pretrained ConvNeXt-Tiny model trained on web images should be
downloaded and placed inside the cloned dinov3 repository folder.
AnS can run without downloading the dinov3 repository and the pretrained model,
but it takes more time because training must be done from scratch.

## Datasets
We provide sample data from Las Vegas in the `sample_data` folder.
This folder contains a subfolder named after the city, which includes separate `train`, `val`, and `test` directories.
Within the city folder, there is also a `data.csv` file that contains the filenames along with
the corresponding translation and rotation parameters.
The supplementary material explains how these parameters can be generated.
The structure of the sample_data folder can be used as a reference for organizing the input data.

## Train
To evaluate the model on the test set from the sample data, run:
```python test.py --keywod test_run```
Similar to training, additional dataset directories can be configured in the train.py file.

To train the model on the sample data, run:
```python train.py --keywod test_run```
New directories can be added in the `train.py` file.

## Test

## Align and segment

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