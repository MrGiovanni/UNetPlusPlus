# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

UNet++ is a new general purpose image segmentation architecture for more accurate image segmentation. UNet++ consists of U-Nets of varying depths whose decoders are densely connected at the same resolution via the redesigned skip pathways, which aim to address two key challenges of the U-Net: 1) unknown depth of the optimal architecture and 2) the unnecessarily restrictive design of skip connections.

## Paper

This repository provides the official Keras implementation of UNet++ in the following papers:

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

## Official implementation

- keras/
- pytorch/

## Other implementation
- [[PyTorch](https://github.com/qubvel/segmentation_models.pytorch)] (by Pavel Yakubovskiy)
- [[PyTorch](https://github.com/4uiiurz1/pytorch-nested-unet)] (by 4ui_iurz1)
- [[PyTorch](https://towardsdatascience.com/biomedical-image-segmentation-unet-991d075a3a4b)] (by Hong Jing)
- [[PyTorch](https://github.com/ZJUGiveLab/UNet-Version)] (by ZJUGiveLab)
- [[PyTorch](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)] (by MontaEllis)
- [[Keras](https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder)] (by Siddhartha)



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

@phdthesis{zhou2021towards,
  title={Towards Annotation-Efficient Deep Learning for Computer-Aided Diagnosis},
  author={Zhou, Zongwei},
  year={2021},
  school={Arizona State University}
}
```

## Acknowledgments

This research has been supported partially by NIH under Award Number R01HL128785, by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH. This is a patent-pending technology.
