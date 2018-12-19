# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

This is an implementation of ["UNet++: A Nested U-Net Architecture for Medical Image Segmentation"](https://arxiv.org/pdf/1807.10165.pdf) in Python and powered by the Keras deep learning framework (Tensorflow as backend). For the first time, a new architecture, called **UNet++** (nested U-Net architecture), is proposed for a more precise segmentation. We introduced the intermediate layers to U-Nets, which naturally form multiple new up-sampling expanding paths of different depths, resulting in an ensemble of U-Nets with a partially shared contracting path.

<p align="center">
  <img src="https://github.com/MrGiovanni/Nested-UNet/blob/master/Figures/fig_UNet%2B%2B.png" width="700"/>
</p>

## License

Detectron is released under the [MIT](https://github.com/MrGiovanni/Nested-UNet/blob/master/LICENSE).

## Citing UNet++

If you use UNet++ in your research, please consider the following BibTeX entry.

```
@incollection{zhou2018unet++,
  title={UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
  pages={3--11},
  year={2018},
  publisher={Springer}
}
```

## Requirements
Python 3.x, Keras 2.2.2, Tensorflow 1.4.1 and other common packages listed in `requirements.txt`.

## Avaliable models:
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [DLA](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf)
 - [UNet++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 
## Avaliable backbones:
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

## Code examples

Train UNet++ structure (`Xnet` in the code):  
```python
from segmentation_models import Unet, Nestnet, Xnet

# prepare data
x, y = ... # range in [0,1]

# prepare model
model = Xnet(backbone_name='resnet34', encoder_weights='imagenet') # biold UNet++
# model = Unet(backbone_name='resnet34', encoder_weights='imagenet') # build U-Net
# model = NestNet(backbone_name='resnet34', encoder_weights='imagenet') # build DLA

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
model.fit(x, y)
```

## TODO
- [x] Add VGG backbone for UNet++
- [x] Add ResNet backbone for UNet++
- [x] Add ResNeXt backbone for UNet++
- [ ] Add DenseNet backbone for UNet++
- [ ] Add Inception backbone for UNet++
- [ ] Add [Tiramisu](https://arxiv.org/pdf/1611.09326.pdf]) and Tiramisu++
- [ ] Add FPN++
- [ ] Add Linknet++
- [ ] Add PSPNet++

## Contacts (Maintainers)

*   Zongwei Zhou, homepage: [zongweiz.com](https://www.zongweiz.com)
*   Md Mahfuzur Rahman Siddiquee, github: [mahfuzmohammad](https://github.com/mahfuzmohammad)
