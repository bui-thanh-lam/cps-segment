# Cross-pseudo Supervised Trasnformer-based Neural Networks for Colorectal Polyp Segmentation

This project is under my graduation thesis at Hanoi University of Science and Technology, and only for research/educational purposes.

## Requirements

- Unbuntu 20.04
- Python 3.7
- Anaconda 4.12.x

## Installation

First, clone this repository:

```
git clone https://github.com/bui-thanh-lam/cps-segment.git
```

and checkout to the root directory of this project.

Then, create a new Anaconda virtual environment with Python 3.7:

```
conda create python=3.7 --name n-cps
conda activate n-cps
```

Install dependencies:

```
pip install -r requirements.txt
```

Download all datasets from this link: https://drive.google.com/drive/folders/1dnpp7xPRWX3-Qw2cmx8Q2OJR5oQw62AI?usp=sharing
and place them to the desired path.

## Usage

1. Train a new model

Command example:

```
python train.py --model_config segformer_b1 --out_dir ../checkpoints/segformer_b1
```

You should learn about the hyperparemters with:

```
python train.py -h
```

2. Evaluate a trained model

Command example:

```
python evaluate.py --model_config segformer_b1 --checkpoint_path ../checkpoints/segformer_b1
```

You should learn about the hyperparemters with:

```
python evaluation.py -h
```

An example of checkpoints can be found in: https://drive.google.com/drive/folders/127WfdW8vw4Sb75bBKGNoiUyEHKISFhIw?usp=sharing

