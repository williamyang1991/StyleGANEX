# StyleGANEX - Official PyTorch Implementation

https://user-images.githubusercontent.com/18130694/224256980-03fb15e7-9858-4300-9d35-7604d03c69f9.mp4

This repository provides the official PyTorch implementation for the following paper:

**StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
[**Project Page**](https://www.mmlab-ntu.com/project/styleganex/) | [**Paper**](#) | [**Supplementary Video**](https://youtu.be/8oK0TXQmxg8) <br>


<a href="http://colab.research.google.com/github/williamyang1991/StyleGANEX/blob/master/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
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
- PyTorch 1.7.1
- Pillow 8.3.1; Matplotlib 3.4.2; opencv-python 4.5.3; Faiss 1.7.1; tqdm 4.61.2; Ninja 1.10.2; dlib 19.24.0


## (1) Inference

### Inference Notebook 
<a href="http://colab.research.google.com/github/williamyang1991/StyleGANEX/blob/master/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 

To help users get started, we provide a Jupyter notebook found in `./inference_playground.ipynb` that allows one to visualize the performance of StyleGANEX.
The notebook will download the necessary pretrained models and run inference on the images found in `./data/`.

### Pre-trained Models

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zGssOxjdklMd_5kdBKV9VkENnS5EXZlx?usp=share_link) or [Hugging Face](https://huggingface.co/PKUWilliamYang/StyleGANEX/tree/main/pretrained_models):

<table>
    <tr>
        <th>Task</th><th>Model</th><th>Description</th>
    </tr>
    <tr>
        <td>Inversion</td><td><a href="https://drive.google.com/file/d/157twOAYihuy_b6l_XrmP7QUOMhkNC2Ii/view?usp=share_link">styleganex_inversion.pt</a></td><td>pre-trained model for StyleGANEX inversion</td>
    </tr>
    <tr>
        <td rowspan="4">Image translation</td><td><a href="https://drive.google.com/file/d/1ewbdY_0fRZS6GIboFcvx6QDBbqHXprvR/view?usp=share_link">styleganex_sr32.pt</a></td><td>pre-trained model specially for 32x face super resolution</td>
    </tr>  
    <tr>
        <td><a href="https://drive.google.com/file/d/1XQ4vp8DB2dSrvQVj3xifSl4sUGMxr4zK/view?usp=share_link">styleganex_sr.pt</a></td><td>pre-trained model for 4x-48x face super resolution</td>
    </tr>   
    <tr>
        <td><a href="https://drive.google.com/file/d/1L3iRp3UE-_Or0NqzUtqZPLQ9hQ_9yax5/view?usp=share_link">styleganex_sketch2face.pt</a></td><td>pre-trained model for skech-to-face translation</td>
    </tr>  
    <tr>
        <td><a href="https://drive.google.com/file/d/1rHC63z1tUX63-56RUc0thhHSVAQr8Cf2/view?usp=share_link">styleganex_mask2face.pt</a></td><td>pre-trained model for parsing map-to-face translation</td>
    </tr>
    <tr>
        <td rowspan="5">Video editing</td><td><a href="https://drive.google.com/file/d/164eD7pafO74xiCCFLzhb56ofOBwIUnpL/view?usp=share_link">styleganex_edit_hair.pt</a></td><td>pre-trained model for hair color editing on videos</td>
    </tr>  
    <tr>
        <td><a href="https://drive.google.com/file/d/1tH-vlpn5THyD-HoOQrYGuUENH49daAFX/view?usp=share_link">styleganex_edit_age.pt</a></td><td>pre-trained model for age editing on videos</td>
    </tr>  
    <tr>
        <td><a href="https://drive.google.com/file/d/16Kth67C1AX3SjZS3030DChB_eHpqGOsn/view?usp=share_link">styleganex_toonify_cartoon.pt</a></td><td>pre-trained Cartoon model for video face toonification</td>
    </tr>  
    <tr>
        <td><a href="https://drive.google.com/file/d/1OkCw4mrrCvTnEfPOeUxjv7yFpzzQ5trf/view?usp=share_link">styleganex_toonify_arcane.pt</a></td><td>pre-trained Arcane model for video face toonification</td>
    </tr>   
    <tr>
        <td><a href="https://drive.google.com/file/d/1_XZjvj-rQvT2q3hiqPZ8gIClMo3ZGVH-/view?usp=share_link">styleganex_toonify_pixar.pt</a></td><td>pre-trained Pixar model for video face toonification</td>
    </tr>   
    <tr>
        <th colspan="2">Supporting model</th><th> </th>
    </tr>
    <tr>
        <td colspan="2"><a href="https://drive.google.com/file/d/1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR/view">faceparsing.pth</a></td><td>BiSeNet for face parsing from <a href="https://github.com/zllrunning/face-parsing.PyTorch">face-parsing.PyTorch</a></td>
    </tr>      
</table>

The downloaded models are suggested to be put into `./pretrained_models/`

### StyleGANEX Inversion

We can embed a face image into the latent space of StyleGANEX to obtain its w+ latent code and the first-layer feature f with `inversion.py`.

```python
python inversion.py --ckpt STYLEGANEX_MODEL_PATH --data_path FACE_IMAGE_PATH
```
The results are saved in the folder `./output/`.
The results contain a reconstructed image `FILE_NAME_inversion.jpg` and a `FILE_NAME_inversion.pt` file.
You can obtain w+ latent code and the first-layer feature f by 
```python 
latents = torch.load('./output/FILE_NAME_inversion.pt')
wplus_hat = latents['wplus'].to(device) # w+
f_hat = [latents['f'][0].to(device)]    # f
```
The `./inference_playground.ipynb` provides some face editing examples based on `wplus_hat` and `f_hat`.

### Image Translation

`image_translation.py` supports face super-resolution, sketch-to-face translation and parsing map-to-face translation

### Video Editing

`video_editing.py` supports video facial attribute editing and video face toonification.



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
