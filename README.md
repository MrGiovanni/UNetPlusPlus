# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

This is an implementation of ["UNet++: A Nested U-Net Architecture for Medical Image Segmentation"]() in Python and powered by the Keras deep learning framework (Tensorflow as backend). For the first time, a new architecture, called **UNet++** (nested U-Net architecture), is proposed for a more precise segmentation. We introduced the intermediate layers to U-Nets, which naturally form multiple new up-sampling expanding paths of different depths, resulting in an ensemble of U-Nets with a partially shared contracting path.

<p align="center">
  <img src="https://github.com/MrGiovanni/Nested-UNet/blob/master/fig_unet%2B%2B.png" width="700"/>
</p>

## License

Detectron is released under the [MIT]().

## Citing UNet++

If you use UNet++ in your research, please use the following BibTeX entry.

```
@inproceedings{zhou2018nest,
  title={UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh and Jianming Liang},
  booktitle={Deep Learning in Medical Image Analysis},
  year={2018}
}
```

## UNet++ Family

### Benchmarks

- Original U-Net ([Ronneberger, 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28))
- ConvNet with bottleneck unit
- ResNet with residual unit
- DenseNet with dense unit

## FCN++ Family

### Benchmarks feature generator backbone

- VGG-16 ([Simonyan, 2014](https://arxiv.org/abs/1409.1556))
- ResNet-101 ([He, 2016](https://arxiv.org/abs/1512.03385))

Additional backbone architectures may be easily implemented.


