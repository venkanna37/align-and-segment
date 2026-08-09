# This code is under preparation. It will be ready soon!

# Align and Segment (AnS): Unsupervised Learning for Building Segmentation From Misaligned Labels
[arXiv Paper](https://arxiv.org/pdf/2607.10841)

This is the official repository for the ECCV 2026 paper "Align and Segment:
Unsupervised Learning for Building Segmentation From Misaligned Labels".
This paper proposed a method for aligning and segmenting buildings from misaligned labels
without using any **golden labels**. The outline of this repository is given as follows.

* [**Requirements**](#requirements)
* [**Datasets and Weights**](#datasets-and-weights)
* [**Train**](#train)
* [**Test**](#test)
* [**Align and segment**](#align-and-segment)
* [**Citing**](#citing)

## Requirements
By installing packages in `requirements.txt`, trainined weights and results on test can be reproduced.

```
conda create -n "ans" python=3.11.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Datasets and Weights
All the derived datasets and check points uploaded to Hugging Face. They are adutomatically downloaded when training and evaluation.
They can also separately download from
[Datasets](https://huggingface.co/datasets/venkanna37/align-and-segment) 
and [Models](https://huggingface.co/venkanna37/align-and-segment).
To avoid redistributing DINOv3 weights and code, [timm](https://timm.fast.ai/)
library used. It automatically download the weights while training and testing.
The weights of decoder of SNet and TNet are shared to reproduce the results.

Follow the below folder structure for custom datasets.
`train.py` and `test.py` scripts automatically download and arrange both weights and datasets in same structure.

### Folder structure
```
datasets                # All datasets folder
└───lasvegas            # Las Vegas Data
    └───train           # Training set
    │   └───images      # Input images
    │   └───labels      # Input labels
    └───val             # Validation set
    │   └───...(same as train)
    └───test            # Test set
        └───...(same as train)

runs 
└───lasvegas_u          # Checkpoints on Las Vegas data with random noise
        └───decoder.pt  # Decoder weights
        └───tnet.pt     # TNet weights
└───lasvegas_b          # Checkpoints on Las Vegas data with systematic noise
        └───...(same as lasvegas_u)
 ```

## Train
To train the model on the `sample_data`, run:
```python train.py --keywod test_run --dataset_name sample_data```
Change the ```--dataset_name``` to `'paris', 'khartoum', 'lasvegas', 'sanjuan', 'rebo'`
to reproduce the training weights on respective dataset. Use `--noise_type` to train on randon (`u`) and systematic (`b`)
noise type when training on `'paris', 'khartoum', 'lasvegas'` datasets. Similarly check `python train.py --help` for more details.


## Test
To evaluate the model on the test set from the sample data, run:
```python test.py --keyword test_run```
The folder names in `runs` folder or in [Hugging Face Model](https://huggingface.co/venkanna37/align-and-segment) are keywords.

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