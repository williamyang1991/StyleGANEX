# StyleGANEX - Official PyTorch Implementation

https://user-images.githubusercontent.com/18130694/224256980-03fb15e7-9858-4300-9d35-7604d03c69f9.mp4

This repository provides the official PyTorch implementation for the following paper:

**StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In ICCV 2023. <br>
[**Project Page**](https://www.mmlab-ntu.com/project/styleganex/) | [**Paper**](https://arxiv.org/abs/2303.06146) | [**Supplementary Video**](https://youtu.be/8oK0TXQmxg8) <br>


<a href="http://colab.research.google.com/github/williamyang1991/StyleGANEX/blob/master/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PKUWilliamYang/StyleGANEX)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=williamyang1991/styleganex)
<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=williamyang1991/styleganex)-->

> **Abstract:** *Recent advances in face manipulation using StyleGAN have produced impressive results. However, StyleGAN is inherently limited to cropped aligned faces at a fixed image resolution it is pre-trained on.  In this paper, we propose a simple and effective solution to this limitation by using dilated convolutions to rescale the receptive fields of shallow layers in StyleGAN, without altering any model parameters.  This allows fixed-size small features at shallow layers to be extended into larger ones that can accommodate variable resolutions, making them more robust in characterizing unaligned faces. To enable real face inversion and manipulation, we introduce a corresponding encoder that provides the first-layer feature of the extended StyleGAN in addition to the latent style code. We validate the effectiveness of our method using unaligned face inputs of various resolutions in a diverse set of face manipulation tasks, including facial attribute editing, super-resolution, sketch/mask-to-face translation, and face toonification.*

**Features**:<br> 
- **Support for Unaligned Faces**: StyleGANEX can manipulate normal field-of-view face images and videos.
- **Compatibility**: StyleGANEX can directly load pre-trained StyleGAN parameters without retraining.
- **Flexible Manipulation**: StyleGANEX retains the style representation and editing ability of StyleGAN.

![overview](https://user-images.githubusercontent.com/18130694/224257328-b6d9bac1-d467-468f-9dba-c89dfed8b931.jpg)

## Updates

- [07/2023] Training code is released.
- [07/2023] The paper is accepted to ICCV 2023 😁!
- [03/2023] Integrated to 🤗 [Hugging Face](https://huggingface.co/spaces/PKUWilliamYang/StyleGANEX). Enjoy the web demo!
- [03/2023] Inference code is released.
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
- Pillow 8.3.1; Matplotlib 3.4.2; opencv-python 4.5.3; tqdm 4.61.2; Ninja 1.10.2; dlib 19.24.0; gradio 3.4

<br/>

## (1) Inference

### Inference Notebook 
<a href="http://colab.research.google.com/github/williamyang1991/StyleGANEX/blob/master/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 

To help users get started, we provide a Jupyter notebook found in `./inference_playground.ipynb` that allows one to visualize the performance of StyleGANEX.
The notebook will download the necessary pretrained models and run inference on the images found in `./data/`.

### Gradio demo
We also provide a UI for testing StyleGANEX that is built with gradio. Running the following command in a terminal will launch the demo:
```
python app_gradio.py
```
This demo is also hosted on [Hugging Face](https://huggingface.co/spaces/PKUWilliamYang/StyleGANEX).

### Pre-trained Models

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zGssOxjdklMd_5kdBKV9VkENnS5EXZlx?usp=share_link),
[Baidu Cloud](https://pan.baidu.com/s/15SZrkvgduUaMI33LJGnEJg?pwd=luck ) (access code: luck)
or [Hugging Face](https://huggingface.co/PKUWilliamYang/StyleGANEX/tree/main/pretrained_models):

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

`image_translation.py` supports face super-resolution, sketch-to-face translation and parsing map-to-face translation.

```python
python image_translation.py --ckpt STYLEGANEX_MODEL_PATH --data_path FACE_INPUT_PATH
```
The results are saved in the folder `./output/`.

Additional notes to consider:
- `--parsing_model_ckpt` (default: `pretrained_models/faceparsing.pth`): path to the pre-trained parsing model
- `--resize_factor` (default: 32): super resolution resize factor
    - For [styleganex_sr.pt](), should be in [4, 48]
    - For [styleganex_sr32.pt](), should be 32
- `--number` (default: 4): output number of multi-modal translation (for sketch/mask-to-face translation task)
- `--use_raw_data` (default: False): 
    - if not specified, apply possible pre-processing to the input data
        - For [styleganex_sr/sr32.pt](), the input face image, e.g., `./data/ILip77SbmOE.png` will be downsampled based on `--resize_factor`. The downsampled image will be also saved in `./output/`.
        - For [styleganex_sketch2face.pt](), no pre-processing will be applied.
        - For [styleganex_mask2face.pt](), the input face image, e.g., `./data/ILip77SbmOE.png` will be transformed into a parsing map. The parsing map and its visualization version will be also saved in `./output/`.
    - if specified, directly load input data without pre-processing
        - For [styleganex_sr/sr32.pt](), the input should be downsampled face images, e.g., `./data/ILip77SbmOE_45x45.png`
        - For [styleganex_sketch2face.pt](), the input should be a one-channel sketch image e.g., `./data/234_sketch.jpg`
        - For [styleganex_mask2face.pt](), the input should be a one-channel parsing map e.g., `./data/ILip77SbmOE_mask.png`


### Video Editing

`video_editing.py` supports video facial attribute editing and video face toonification.

```python
python video_editing.py --ckpt STYLEGANEX_MODEL_PATH --data_path FACE_INPUT_PATH
```
The results are saved in the folder `./output/`.

Additional notes to consider:
- `--data_path`: the input can be either an image or a video.
- `--scale_factor`: for attribute editing task ([styleganex_edit_hair/age]()), control the editing degree.

<br/>

## (2) Training 

### Preparing your Data
- As with pSp, we provide support for numerous datasets and experiments (encoding, translation, etc.).
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
    - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment. 
    - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
- If you wish to experiment with your own dataset, you can simply make the necessary adjustments in 
    - `data_configs.py` to define your data paths.
    - `transforms_configs.py` to define your own data transforms.
    
As an example, assume we wish to run encoding using ffhq (`dataset_type=ffhq_encode`). 
We first go to `configs/paths_config.py` and define:
``` 
dataset_paths = {
    'ffhq': '/path/to/ffhq/realign320x320'
    'ffhq_test': '/path/to/ffhq/realign320x320_test'
}
```
The transforms for the experiment are defined in the class `EncodeTransforms` in `configs/transforms_config.py`.   
Finally, in `configs/data_configs.py`, we define:
``` 
DATASETS = {
   'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['ffhq_test'],
        'test_target_root': dataset_paths['ffhq_test'],
    },
}
``` 
When defining our datasets, we will take the values in the above dictionary.

The 1280x1280 ffhq images can be obtain by the modified script of official ffhq:
- Download the in-the-wild images with `python script/download_ffhq1280.py --wilds`
- Reproduce the aligned 1280×1280 images wiht `python script/download_ffhq1280.py --align`
- 320x320 ffhq images can be obtained by setting `output_size=320, transform_size=1280` in Line 272 of download_ffhq1280.py

### Downloading supporting models
Please download the pre-trained models to support the training of StyleGANEX
| Path | Description
| :--- | :----------
|[original_stylegan](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)| StyleGAN trained with the FFHQ dataset
|[toonify_model](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b) | StyleGAN finetuned on cartoon dataset for image toonification ([cartoon](https://drive.google.com/file/d/1w7BJDiSw5_ybelv7jL_Jeu1T-oWEWUmH/view?usp=drive_link), [pixar](https://drive.google.com/file/d/1phftRYbsp34pL5Yqapz3c_Wv0G4L0vy2/view?usp=drive_link), [arcane](https://drive.google.com/file/d/1HysdpShIAbHtf6T9R-hVULjUShgA5AL1/view?usp=drive_link))
|[original_psp_encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing)  | pSp trained with the FFHQ dataset for StyleGAN inversion.
|[pretrained_encoder](https://drive.google.com/file/d/1RkQbKZUoTSBKRPwAcpXCPmLExIIs84eO/view?usp=drive_link)  | StyleGANEX encoder pretrained with the synthetic data for StyleGAN inversion.
|[styleganex_encoder](https://drive.google.com/file/d/157twOAYihuy_b6l_XrmP7QUOMhkNC2Ii/view?usp=share_link)  | StyleGANEX encoder trained with the FFHQ dataset for StyleGANEX inversion.
|[editing_vector](https://drive.google.com/drive/folders/1zGssOxjdklMd_5kdBKV9VkENnS5EXZlx?usp=share_link)  | Editing vectors for editing face attributes ([age](https://drive.google.com/file/d/1j2373q_xETMJoJGaLQriLrzzzpTQdgWD/view?usp=drive_link), [hair color](https://drive.google.com/file/d/1qCelrIaF4GieKqjwvi5siQbYwKTzqK4M/view?usp=drive_link))
|[augmentation_vector](https://drive.google.com/file/d/1cOspde7cWY7AYsTQ0X4_DXRfv_hdZ2Y0/view?usp=drive_link)  | Editing vectors for data augmentation

The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  

### Training styleganex

Note: Our default code is a CPU-compatible version. You can switch to a more efficient version by using cpp extention. 
To do so, please change `models.stylegan2.op` to `models.stylegan2.op_old`
https://github.com/williamyang1991/StyleGANEX/blob/73b580cc7eb757e36701c094456e9ee02078d03e/models/stylegan2/model.py#L8

#### Training the styleganex encoder
First pretrain encoder on synthetic 1024x1024 images. You can download our pretrained encoder [here](https://drive.google.com/file/d/1RkQbKZUoTSBKRPwAcpXCPmLExIIs84eO/view?usp=drive_link)
```
python scripts/pretrain.py \
--exp_dir=/path/to/experiment \
--ckpt=/path/to/original_psp_encoder \
--max_steps=2000
```
Then finetune encoder on real 1280x1280 ffhq images based on the [pretrained encoder](https://drive.google.com/file/d/1RkQbKZUoTSBKRPwAcpXCPmLExIIs84eO/view?usp=drive_link)
```
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/pretrained_encoder \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--val_interval=2500 \
--save_interval=50000 \
--start_from_latent_avg \
--id_lambda=0.1 \
--w_norm_lambda=0.001 \
--affine_augment \
--random_crop \
--crop_face
```
#### Sketch to Face
```
python scripts/train.py \
--dataset_type=ffhq_sketch_to_face \
--exp_dir=/path/to/experiment \
--stylegan_weights=/path/to/original_stylegan \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_latent_avg \
--w_norm_lambda=0.005 \
--affine_augment \
--random_crop \
--crop_face \
--use_skip \
--skip_max_layer=1 \
--label_nc=1 \
--input_nc=1 \
--use_latent_mask
```
#### Segmentation Map to Face
```
python scripts/train.py \
--dataset_type=ffhq_seg_to_face \
--exp_dir=/path/to/experiment \
--stylegan_weights=/path/to/original_stylegan \
--max_steps=100000 \
--workers=8 \
--batch_size=8 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_latent_avg \
--w_norm_lambda=0.005 \
--affine_augment \
--random_crop \
--crop_face \
--use_skip \
--skip_max_layer=2 \
--label_nc=19 \
--input_nc=19 \
--use_latent_mask 
```
#### Super Resolution
``` 
python scripts/train.py \
--dataset_type=ffhq_super_resolution \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/styleganex_encoder \
--max_steps=100000 \
--workers=4 \
--batch_size=4 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_latent_avg \
--adv_lambda=0.1 \
--affine_augment \
--random_crop \
--crop_face \
--use_skip \
--skip_max_layer=4 \
--resize_factors=8
```
For one model supporting multiple resize factors, set `--skip_max_layer=2` and `--resize_factors=1,2,4,8,16`
#### Video Editing
```
python scripts/train.py \
--dataset_type=ffhq_edit \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/styleganex_encoder \
--max_steps=100000 \
--workers=2 \
--batch_size=2 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_latent_avg \
--adv_lambda=0.1 \
--tmp_lambda=30 \
--affine_augment \
--crop_face \
--use_skip \
--skip_max_layer=7 \
--editing_w_path=/path/to/editing_vector \
--direction_path=/path/to/augmentation_vector \
--use_att=1 \
--generate_training_data
```
#### Video Toonification
```
python scripts/train.py \
--dataset_type=toonify \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/styleganex_encoder \
--max_steps=55000 \
--workers=2 \
--batch_size=2 \
--val_interval=2500 \
--save_interval=10000 \
--start_from_latent_avg \
--adv_lambda=0.1 \
--tmp_lambda=30 \
--affine_augment \
--crop_face \
--use_skip \
--skip_max_layer=7 \
--toonify_weights=/path/to/toonify_model
```

### Additional Notes
- See `options/train_options.py` for all training-specific flags. 
- If you wish to generate images from segmentation maps, please specify `--label_nc=N`  and `--input_nc=N` where `N` 
is the number of semantic categories. 
- Similarly, for generating images from sketches, please specify `--label_nc=1` and `--input_nc=1`.
- Specifying `--label_nc=0` (the default value), will directly use the RGB colors as input.

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
@inproceedings{yang2023styleganex,
 title = {StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces},
 author = {Yang, Shuai and Jiang, Liming and Liu, Ziwei and and Loy, Chen Change},
 booktitle = {ICCV},
 year = {2023},
}
```

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [VToonify](https://github.com/williamyang1991/VToonify).
