# StyleGANEX - Official PyTorch Implementation

https://user-images.githubusercontent.com/18130694/224256980-03fb15e7-9858-4300-9d35-7604d03c69f9.mp4

This repository provides the official PyTorch implementation for the following paper:

**StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
[**Project Page**](https://www.mmlab-ntu.com/project/styleganex/) | [**Paper**](#) | [**Supplementary Video**](https://youtu.be/8oK0TXQmxg8) <br>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=williamyang1991/styleganex)
<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=williamyang1991/styleganex)-->

> **Abstract:** *Recent advances in face manipulation using StyleGAN have produced impressive results. However, StyleGAN is inherently limited to cropped aligned faces at a fixed image resolution it is pre-trained on.  In this paper, we propose a simple and effective solution to this limitation by using dilated convolutions to rescale the receptive fields of shallow layers in StyleGAN, without altering any model parameters.  This allows fixed-size small features at shallow layers to be extended into larger ones that can accommodate variable resolutions, making them more robust in characterizing unaligned faces. To enable real face inversion and manipulation, we introduce a corresponding encoder that provides the first-layer feature of the extended StyleGAN in addition to the latent style code. We validate the effectiveness of our method using unaligned face inputs of various resolutions in a diverse set of face manipulation tasks, including facial attribute editing, super-resolution, sketch/mask-to-face translation, and face toonification.*

**Features**:<br> 
- **Support for Unaligned Faces**: StyleGANEX can manipulate normal field-of-view face images and videos.
- **Compatibility**: StyleGANEX can directly load pre-trained StyleGAN parameters without retraining.
- **Flexible Manipulation**: StyleGANEX retains the style representation and editing ability of StyleGAN.

![overview](https://user-images.githubusercontent.com/18130694/224257328-b6d9bac1-d467-468f-9dba-c89dfed8b931.jpg)

## Updates

- [03/2023] This website is created.


## Installation

**Clone this repo:**
```bash
git clone https://github.com/williamyang1991/StyleGANEX.git
cd StyleGANEX
```
**Dependencies:**

We have tested on:
- CUDA 10.1
- PyTorch 1.7.0
- Pillow 8.3.1; Matplotlib 3.3.4; opencv-python 4.5.3; Faiss 1.7.1; tqdm 4.61.2; Ninja 1.10.2


## (1) Inference

Comming soon. We are cleaning our code. :)

## (2) Training 

Code will be released upon the publication of paper.

## (3) Results

Overview of StyleGANEX inversion and facial attribute/style editing on unaligned faces:

![result](https://user-images.githubusercontent.com/18130694/224259844-c9b37f4f-c786-48cd-a92f-121606b14b36.jpg)

Video facial attribute editing:

https://user-images.githubusercontent.com/18130694/224287063-7465a301-4d11-4322-819a-59d548308ce1.mp4

<br/>

Video face toonification:

https://user-images.githubusercontent.com/18130694/224287136-7e5ce82d-664f-4a23-8ed3-e7005efb3b24.mp4

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@InProceedings{yang2023styleganex,
 title = {StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces},
 author = {Yang, Shuai and Jiang, Liming and Liu, Ziwei and and Loy, Chen Change},
 year = {2023},
}
```

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [VToonify](https://github.com/williamyang1991/VToonify).
