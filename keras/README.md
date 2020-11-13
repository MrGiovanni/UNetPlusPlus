# UNet++: Official Keras Implementation

This repository provides the official **Keras** implementation of UNet++ in the following papers:

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
 - [U-Net](https://arxiv.org/abs/1505.04597)
 - [DLA](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf) (UNet+)
 - **[UNet++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)**
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 
### 2. Available backbones
| Backbone model      |Name| Weights    |
|---------------------|:--:|:------------:|
| VGG16               |`vgg16`| `imagenet` |
| VGG19               |`vgg19`| `imagenet` |
| ResNet18            |`resnet18`| `imagenet` |
| ResNet34            |`resnet34`| `imagenet` |
| ResNet50            |`resnet50`| `imagenet`<br>`imagenet11k-places365ch` |
| ResNet101           |`resnet101`| `imagenet` |
| ResNet152           |`resnet152`| `imagenet`<br>`imagenet11k` |
| ResNeXt50           |`resnext50`| `imagenet` |
| ResNeXt101          |`resnext101`| `imagenet` |
| DenseNet121         |`densenet121`| `imagenet` |
| DenseNet169         |`densenet169`| `imagenet` |
| DenseNet201         |`densenet201`| `imagenet` |
| Inception V3        |`inceptionv3`| `imagenet` |
| Inception ResNet V2 |`inceptionresnetv2`| `imagenet` |

## How to use UNet++

### 1. Requirements
Python 3.x, Keras 2.2.2, Tensorflow 1.4.1 and other common packages listed in `requirements.txt`.

### 2. Installation

```bash
git clone https://github.com/MrGiovanni/UNetPlusPlus.git
cd UNetPlusPlus/keras
pip install -r requirements.txt
git submodule update --init --recursive
```

### 3. Running the scripts

#### Application 1: [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018)
```bash
CUDA_VISIBLE_DEVICES=0 python DSB2018_application.py --run 1 \
                                                     --arch Xnet \
                                                     --backbone vgg16 \
                                                     --init random \
                                                     --decoder transpose \
                                                     --input_rows 96 \
                                                     --input_cols 96 \
                                                     --input_deps 3 \
                                                     --nb_class 1 \
                                                     --batch_size 2048 \
                                                     --weights None \
                                                     --verbose 1
```
#### Application 2: [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)

#### Application 3: [Polyp Segmentation (ASU-Mayo)](https://polyp.grand-challenge.org/databases/)

#### Application 4: [Lung Image Database Consortium image collection (LIDC-IDRI)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

#### Application 5: [Multiparametric Brain Tumor Segmentation (BRATS 2013)](https://www.smir.ch/BRATS/Start2013)
```bash
CUDA_VISIBLE_DEVICES=0 python BRATS2013_application.py --run 1 \
                                                     --arch Xnet \
                                                     --backbone vgg16 \
                                                     --init random \
                                                     --decoder transpose \
                                                     --input_rows 256 \
                                                     --input_cols 256 \
                                                     --input_deps 3 \
                                                     --nb_class 1 \
                                                     --batch_size 2048 \
                                                     --weights None \
                                                     --verbose 1
```

## Code examples for your own data

Train a UNet++ structure (`Xnet` in the code):  
```python
from segmentation_models import Unet, Nestnet, Xnet

# prepare data
x, y = ... # range in [0,1], the network expects input channels of 3

# prepare model
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
# model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
# model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
model.fit(x, y)
```

## To do
- [x] Add VGG backbone for UNet++
- [x] Add ResNet backbone for UNet++
- [x] Add ResNeXt backbone for UNet++
- [ ] Add DenseNet backbone for UNet++
- [ ] Add Inception backbone for UNet++
- [ ] Add [Tiramisu](https://arxiv.org/pdf/1611.09326.pdf]) and Tiramisu++
- [ ] Add FPN++
- [ ] Add Linknet++
- [ ] Add PSPNet++

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

This repository has been built upon [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models). We appreciate the effort of Pavel Yakubovskiy for providing well-organized segmentation models to the community. This research has been supported partially by NIH under Award Number R01HL128785, by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH. This is a patent-pending technology.
