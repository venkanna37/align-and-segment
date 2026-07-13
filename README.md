# This code is under preparation. It will be ready soon!

# Align and Segment (AnS): Unsupervised Learning for Building Segmentation From Misaligned Labels

This is the official repository for the ECCV 2026 paper "Align and Segment:
Unsupervised Learning for Building Segmentation From Misaligned Labels".
This paper proposed a method for aligning and segmenting buildings from misaligned labels
without using any **golden labels**. The outline of this repository is given as follows.

* [**Requirements**](#requirements)
* [**Datasets**](#datasets)
* [**Train**](#train)
* [**Test**](#test)
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
To test the code, we provided sythetic sample data from Las Vegas
in the `sample_data/vegas` folder. For the trainining, validation and test sets,
we provided the golden labels  and transformation parameters for generating misaligned labels in `data.csv`.
To reproduce the results, we provided sythetic datasets from three cities in this
[link](https://sid.erda.dk/sharelink/fvQxXCQzU6). The folder structure of these datasets is
the same as that of the sample data.
We also provided the dataset used for qualitative evaluation in the same
[link](https://sid.erda.dk/sharelink/fvQxXCQzU6).
The supplementary material of our paper explains how all these datasets were generated.
The real-world dataset, ReBO, can be downloaded by following the instructions
in the [DragOSM](https://github.com/likaiucas/DragOSM) repository. Please read the dataset section
in our paper about how we split and train on ReBO dataset.

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