# Predict-then-Interpolate

This repository contains the code and data for our paper:

*Predict, then Interpolate: A Simple Algorithm to Learn Stable Classifiers*. Yujia Bao, Shiyu Chang and Regina Barzilay.

If you find this work useful and use it on your own research, please cite our paper.

## Overview

Our goal is to learn correlations that are stable across different training environments. Our algorithm consists of three stages:
1. For each training environment <img src="https://render.githubusercontent.com/render/math?math=E_i">, train a classifier <img src="https://render.githubusercontent.com/render/math?math=f_i">.
2. For each pair of training environments <img src="https://render.githubusercontent.com/render/math?math=E_i"> and <img src="https://render.githubusercontent.com/render/math?math=E_j">, use the classifier <img src="https://render.githubusercontent.com/render/math?math=f_i"> to partition <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j = E_j^{i\checkmark} \cup E_j^{i\times}">,
where <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j^{i\checkmark}"> contains examples that are predicted correctly and <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j^{i\times}"> contains examples that are misclassified by <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=f_i">.
3. Train the final model by minimizing the worst-case risk over all interpolations of the partitions.

<p align="center">
<img src="assets/toy.png" width=100% />
</p>

## Data
#### Download

We ran experiments on a total of 4 datasets. MNIST and CelebA can be directly downloaded from the PyTorch API. For beer review and ASK2ME, you may download our processed data [here](https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip).

### Quickstart
`.bin/` contains all the scripts for running the baselines and our algorithm.

## Code
`src/main.py` is our main file.
- `src/train_utils` loads the training algorithm specified by the method argument.
- `src/data_utils` loads the dataset specified by the dataset argument.
- `src/model_utils` loads the network specified by the method and the dataset arguments.
- `src/training/` contains the training and testing routine for all methods.
- `src/data/` contains the data pre-processing and loading pipeline for different datasets.
- `src/model/` contains the networks that we used for different datasets.

## Dependencies
`package-list.txt` contains all the packages that are related to the project.
To install them, simply create a new [conda](https://docs.conda.io/en/latest/) environment and type
```
conda install --file package-list.txt
```

