## SCOMatch: Alleviating Overtrusting \\in Open-set Semi-supervised Learning

This is an PyTorch implementation of SCOMatch (ECCV2024).
This implementation is based on [OpenMatch](https://github.com/VisionLearningGroup/OP_Match).



## Requirements
- python 3.6+
- torch 1.10
- torchvision 0.11.1
- wandb 0.15.8
- numpy
- tqdm
- sklearn

## Usage

### Dataset Preparation
This repository needs CIFAR10, CIFAR100, TinyImageNet and ImageNet-30 to train a model.

- CIFAR10, CIFAR100 will be downloaded automatically.
- Follow [CSI](https://github.com/alinlab/CSI) to prepare ImageNet-30.

```
mkdir data
ln -s path_to_each_dataset ./data/.

## unzip filelist for imagenet_30 experiments.
unzip files.zip
```

All datasets are supposed to be under ./data.

### Environment

Please follow [OpenMatch](https://github.com/VisionLearningGroup/OP_Match) to set up the python environments.

### Train

Please use the scripts in `./scripts` for the experiments on each dataset.

### Acknowledgement
This repository depends a lot on [Pytorch-FixMatch](https://github.com/kekmodel/FixMatch-pytorch) and [OpenMatch](https://github.com/VisionLearningGroup/OP_Match). 
Thanks for sharing the great code bases!

### Reference
If you consider using this code or its derivatives, please consider citing:

```
@inproceedings{wang2024scomatch,
  title={SCOMatch: Alleviating Overtrusting in Open-set Semi-supervised Learning},
  author={Wang, Zerun and Xiang, Liuyu and Huang, Lang and Mao, Jiafeng and Xiao, Ling and Yamasaki, Toshihiko},
  booktitle={European Conference on Computer Vision},
  pages={217--233},
  year={2024},
  organization={Springer}
}
```

