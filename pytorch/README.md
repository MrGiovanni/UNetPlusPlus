# UNet++: Official PyTorch Implementation

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the official **PyTorch** implementation of UNet++ in the following papers:

**UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation** <br/>
[Zongwei Zhou](https://www.zongweiz.com), [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad), [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/), and [Jianming Liang](https://chs.asu.edu/jianming-liang) <br/>
Arizona State University <br/>
IEEE Transactions on Medical Imaging ([TMI](https://ieee-tmi.org/)) <br/>
[paper](https://arxiv.org/abs/1912.05074) | [code](https://github.com/MrGiovanni/Nested-UNet)

**UNet++: A Nested U-Net Architecture for Medical Image Segmentation** <br/>
[Zongwei Zhou](https://www.zongweiz.com), [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad), [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/), and [Jianming Liang](https://chs.asu.edu/jianming-liang) <br/>
Arizona State University <br/>
Deep Learning in Medical Image Analysis ([DLMIA](https://cs.adelaide.edu.au/~dlmia4/)) 2018. **(Oral)** <br/>
[paper](https://arxiv.org/abs/1807.10165) | [code](https://github.com/MrGiovanni/Nested-UNet) | [slides](https://docs.wixstatic.com/ugd/deaea1_1d1e512ebedc4facbb242d7a0f2b7a0b.pdf) | [poster](https://docs.wixstatic.com/ugd/deaea1_993c14ef78f844c88a0dae9d93e4857c.pdf) | [blog](https://zhuanlan.zhihu.com/p/44958351)


## What is in this repository

### 1. Available architectures
 - [3D U-Net](https://arxiv.org/abs/1505.04597)
 - **[3D UNet++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)**


## How to use UNet++

### 1. Requirements
Linux, Python 3.7+, PyTorch 1.6+

### 2. Installation

- Create virtual environment. [Here is a quick how-to for Ubuntu](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
- Install [PyTorch](https://pytorch.org/get-started/locally/)
- Install nnUNet++ as below
```
git clone https://github.com/MrGiovanni/UNetPlusPlus.git
cd UNetPlusPlus/pytorch
pip install git+https://github.com/MIC-DKFZ/batchgenerators.git
pip install -e .
```
- Set a few environment variables. Please follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)

### 3. Running the scripts

#### Application 1: Liver and tumour segmentation in Medical Segmentation Decathlon ([download models](https://drive.google.com/drive/folders/1mY6nOoL9ddHyFqNIKqBN9l3GhPegye8H?usp=sharing))

| experiment      | Liver 1_Dice (val) | Liver 2_Dice (val) | Liver 1_Dice (test) | Liver 2_Dice (te) | 
|-----------------|:------------------:|:------------------:|:-----------------:|:-----------------:|
| U-Net (reported)               | 95.71 |  63.72  | 95.75 | 75.97 |
| U-Net (reproduced)          | **96.18** | 63.37 | - | - |
| UNet++ (1st branch)         | 89.96 | 38.03 | - | - |
| UNet++ (2nd branch)        | 94.68 | 59.66 | - | - |
| UNet++ (3rd branch)         | 95.78 |  65.83  | - | - |
| UNet++ (4th branch)         | 96.03 | 66.02   | - | - |
| UNet++ (5th branch)         | 96.11 |  **66.25**  | 95.21 | **76.53** |
| UNet++ (average)            |  95.80 |  65.60  | - | - |

- Run the following command to train on Task003_Liver:

```bash
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 Task003_Liver $FOLD
done
```

## Citation
If you use UNet++ for your research, please cite our papers:
```
@article{zhou2019unetplusplus,
  title={UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal={IEEE Transactions on Medical Imaging},
  year={2019},
  publisher={IEEE}
}

@incollection{zhou2018unetplusplus,
  title={Unet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
  pages={3--11},
  year={2018},
  publisher={Springer}
}
```

## Acknowledgments

We thank [Shivam Bajpai](https://github.com/sbajpai2) for implementating UNet++ and participating in the Medical Segmentation Decathlon challenge. This repository has been built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). We appreciate the effort of Fabian Isensee for providing competitive segmentation framework to the community. This research has been supported partially by NIH under Award Number R01HL128785, by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH. This is a patent-pending technology.
