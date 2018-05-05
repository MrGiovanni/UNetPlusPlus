# Nested Neural Networks for Medical Image Segmentation

This is an implementation of ["Nested Neural Networks for Medical Image Segmentation"](https://openreview.net/pdf?id=ryPLSWnsM) in Python and powered by the Keras deep learning framework (Tensorflow as backend). For the first time, a new architecture, called **Nest-Net** (nested ensemble nets), is proposed for a more precise segmentation. We introduced the intermediate layers to U-Nets, which naturally form multiple new up-sampling expanding paths of different depths, resulting in an ensemble of U-Nets with a partially shared contracting path.

<a align="center">
  <img src="https://github.com/MrGiovanni/Nest-Net/blob/master/fig-network-architecture.png" width="600"/>
</a>

## License

Detectron is released under the [MIT]().

## Citing Nest-Net

If you use Nest-Net in your research, please use the following BibTeX entry.

```
@inproceedings{zhou2018nestnet,
  title={Nested Neural Networks for Medical Image Segmentation},
  author={Zongwei Zhou, Md Mahfuzur Rahman Siddiquee and Jianming Liang},
  booktitle={International Conference on Medical Imaging with Deep Learning (MIDL)},
  year={2018}
}
```

## Nested U-Net Family

### Benchmarks

- Original U-Net ([Ronneberger, 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28))
- ConvNet with bottleneck unit
- ResNet with residual unit
- DenseNet with dense unit

## Nested FCN Family

using the following backbone network architectures:

- VGG-16 ([Simonyan, 2014](https://arxiv.org/abs/1409.1556))
- ResNet-101 ([He, 2016](https://arxiv.org/abs/1512.03385))

Additional backbone architectures may be easily implemented.


