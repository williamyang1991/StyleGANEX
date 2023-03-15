from __future__ import annotations
import numpy as np
import gradio as gr

import os
import pathlib
import gc
import torch
import dlib
import cv2
import PIL
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from argparse import Namespace
from datasets import augmentations
from huggingface_hub import hf_hub_download
from scripts.align_all_parallel import align_face
from latent_optimization import latent_optimization
from utils.inference_utils import save_image, load_image, visualize, get_video_crop_parameter, tensor2cv2, tensor2label, labelcolormap
from models.psp import pSp
from models.bisenet.model import BiSeNet
from models.stylegan2.model import Generator

class Model():
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.task_name = None
        self.editing_w = None
        self.pspex = None
        self.landmarkpredictor = dlib.shape_predictor(hf_hub_download('PKUWilliamYang/VToonify', 'models/shape_predictor_68_face_landmarks.dat'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.maskpredictor = BiSeNet(n_classes=19)
        self.maskpredictor.load_state_dict(torch.load(hf_hub_download('PKUWilliamYang/VToonify', 'models/faceparsing.pth'), map_location='cpu'))
        self.maskpredictor.to(self.device).eval()
        self.parameters = {}
        self.parameters['inversion'] = {'path':'pretrained_models/styleganex_inversion.pt', 'image_path':'./data/ILip77SbmOE.png'}
        self.parameters['sr-32'] = {'path':'pretrained_models/styleganex_sr32.pt', 'image_path':'./data/pexels-daniel-xavier-1239291.jpg'}
        self.parameters['sr'] = {'path':'pretrained_models/styleganex_sr.pt', 'image_path':'./data/pexels-daniel-xavier-1239291.jpg'}
        self.parameters['sketch2face'] = {'path':'pretrained_models/styleganex_sketch2face.pt', 'image_path':'./data/234_sketch.jpg'}
        self.parameters['mask2face'] = {'path':'pretrained_models/styleganex_mask2face.pt', 'image_path':'./data/540.jpg'}
        self.parameters['edit_age'] = {'path':'pretrained_models/styleganex_edit_age.pt', 'image_path':'./data/390.mp4'}
        self.parameters['edit_hair'] = {'path':'pretrained_models/styleganex_edit_hair.pt', 'image_path':'./data/390.mp4'}
        self.parameters['toonify_pixar'] = {'path':'pretrained_models/styleganex_toonify_pixar.pt', 'image_path':'./data/pexels-anthony-shkraba-production-8136210.mp4'}
        self.parameters['toonify_cartoon'] = {'path':'pretrained_models/styleganex_toonify_cartoon.pt', 'image_path':'./data/pexels-anthony-shkraba-production-8136210.mp4'}
        self.parameters['toonify_arcane'] = {'path':'pretrained_models/styleganex_toonify_arcane.pt', 'image_path':'./data/pexels-anthony-shkraba-production-8136210.mp4'}        
        self.print_log = True
        self.editing_dicts = torch.load(hf_hub_download('PKUWilliamYang/StyleGANEX', 'direction_dics.pt'))
        self.generator = Generator(1024, 512, 8)
        self.model_type = None
        self.error_info = 'Error: no face detected! \
                               StyleGANEX uses dlib.get_frontal_face_detector but sometimes it fails to detect a face. \
                               You can try several times or use other images until a face is detected, \
                               then switch back to the original image.'
        
    def load_model(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pspex is not None:
            del self.pspex
        torch.cuda.empty_cache()
        gc.collect()
        path = self.parameters[task_name]['path']
        local_path = hf_hub_download('PKUWilliamYang/StyleGANEX', path)
        ckpt = torch.load(local_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = local_path
        opts['device'] = self.device
        opts = Namespace(**opts)
        self.pspex = pSp(opts, ckpt).to(self.device).eval()
        self.pspex.latent_avg = self.pspex.latent_avg.to(self.device)
        if 'editing_w' in ckpt.keys():
            self.editing_w = ckpt['editing_w'].clone().to(self.device)   
        self.task_name = task_name  
        torch.cuda.empty_cache()
        gc.collect()

    def load_G_model(self, model_type: str) -> None:
        if model_type == self.model_type:
            return
        torch.cuda.empty_cache()
        gc.collect()
        local_path = hf_hub_download('rinong/stylegan-nada-models', model_type+'.pt')
        self.generator.load_state_dict(torch.load(local_path, map_location='cpu')['g_ema'], strict=False)
        self.generator.to(self.device).eval()
        self.model_type = model_type  
        torch.cuda.empty_cache()
        gc.collect()
        
    def tensor2np(self, img):
        tmp = ((img.cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        return tmp

    def process_sr(self, input_image: str, resize_scale: int, model: str) -> list[np.ndarray]:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the photo.'
            
        if input_image is None:
            #return [false_image, false_image], 'Error: fail to load empty file.'
            raise gr.Error("Error: fail to load empty file.")
        frame = cv2.imread(input_image)
        if frame is None:
            #return [false_image, false_image], 'Error: fail to load the image.'
            raise gr.Error("Error: fail to load the image.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            
        if model is None or model == 'SR for 32x':
            task_name = 'sr-32'
            resize_scale = 32
        else:
            task_name = 'sr'
            
        with torch.no_grad():
            paras = get_video_crop_parameter(frame, self.landmarkpredictor)
            if paras is None:
                #return [false_image, false_image], info
                raise gr.Error(self.error_info)
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            x1 = PIL.Image.fromarray(np.uint8(frame))
            x1 = augmentations.BilinearResize(factors=[resize_scale//4])(x1)
            x1_up = x1.resize((W, H))
            x2_up = align_face(np.array(x1_up), self.landmarkpredictor)
            if x2_up is None:
                #return [false_image, false_image], 'Error: no face detected! Please retry or change the photo.'
                raise gr.Error(self.error_info)
            x1_up = transforms.ToTensor()(x1_up).unsqueeze(dim=0).to(self.device) * 2 - 1
            x2_up = self.transform(x2_up).unsqueeze(dim=0).to(self.device)
            if self.print_log: print('image loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            y_hat = torch.clamp(self.pspex(x1=x1_up, x2=x2_up, use_skip=self.pspex.opts.use_skip, resize=False), -1, 1)

        return [self.tensor2np(x1_up[0]), self.tensor2np(y_hat[0])]
    
    
    def process_s2f(self, input_image: str, seed: int) -> np.ndarray:
        task_name = 'sketch2face'
        with torch.no_grad():
            x1 = transforms.ToTensor()(PIL.Image.open(input_image)).unsqueeze(0).to(self.device)
            if x1.shape[2] > 513:
                x1 = x1[:,:,(x1.shape[2]//2-256)//8*8:(x1.shape[2]//2+256)//8*8]
            if x1.shape[3] > 513:
                x1 = x1[:,:,:,(x1.shape[3]//2-256)//8*8:(x1.shape[3]//2+256)//8*8]
            x1 = x1[:,0:1] # uploaded files will be transformed to 3-channel RGB image!
            if self.print_log: print('image loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            self.pspex.train()
            torch.manual_seed(seed)
            y_hat = self.pspex(x1=x1, resize=False, latent_mask=[8,9,10,11,12,13,14,15,16,17], use_skip=self.pspex.opts.use_skip,
                                  inject_latent= self.pspex.decoder.style(torch.randn(1, 512).to(self.device)).unsqueeze(1).repeat(1,18,1) * 0.7)  
            y_hat = torch.clamp(y_hat, -1, 1)
            self.pspex.eval()
        return self.tensor2np(y_hat[0])
    
    def process_m2f(self, input_image: str, input_type: str, seed: int) -> list[np.ndarray]:
        #false_image = np.zeros((256,256,3), np.uint8)
        if input_image is None:
            raise gr.Error('Error: fail to load empty file.' )
            #return [false_image, false_image], 'Error: fail to load empty file.'        
        task_name = 'mask2face'
        with torch.no_grad():
            if input_type == 'parsing mask':
                x1 = PIL.Image.open(input_image).getchannel(0) # uploaded files will be transformed to 3-channel RGB image!
                x1 = augmentations.ToOneHot(19)(x1)
                x1 = transforms.ToTensor()(x1).unsqueeze(dim=0).float().to(self.device)
                #print(x1.shape)
            else:
                frame = cv2.imread(input_image)
                if frame is None:
                    #return [false_image, false_image], 'Error: fail to load the image.' 
                    raise gr.Error('Error: fail to load the image.' )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                paras = get_video_crop_parameter(frame, self.landmarkpredictor)        
                if paras is None:
                    #return [false_image, false_image], 'Error: no face detected! Please retry or change the photo.'
                    raise gr.Error(self.error_info)
                h,w,top,bottom,left,right,scale = paras
                H, W = int(bottom-top), int(right-left)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]        
                # convert face image to segmentation mask
                x1 = self.to_tensor(frame).unsqueeze(0).to(self.device)
                # upsample image for precise segmentation
                x1 = F.interpolate(x1, scale_factor=2, mode='bilinear') 
                x1 = self.maskpredictor(x1)[0]
                x1 = F.interpolate(x1, scale_factor=0.5).argmax(dim=1)
                x1 = F.one_hot(x1, num_classes=19).permute(0, 3, 1, 2).float().to(self.device)   
                
            if x1.shape[2] > 513:
                x1 = x1[:,:,(x1.shape[2]//2-256)//8*8:(x1.shape[2]//2+256)//8*8]
            if x1.shape[3] > 513:
                x1 = x1[:,:,:,(x1.shape[3]//2-256)//8*8:(x1.shape[3]//2+256)//8*8]

            x1_viz = (tensor2label(x1[0], 19) / 192 * 256).astype(np.uint8)

            if self.print_log: print('image loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            self.pspex.train()
            torch.manual_seed(seed)
            y_hat = self.pspex(x1=x1, resize=False, latent_mask=[8,9,10,11,12,13,14,15,16,17], use_skip=self.pspex.opts.use_skip,
                                  inject_latent= self.pspex.decoder.style(torch.randn(1, 512).to(self.device)).unsqueeze(1).repeat(1,18,1) * 0.7)  
            y_hat = torch.clamp(y_hat, -1, 1)
            self.pspex.eval()
        return [x1_viz, self.tensor2np(y_hat[0])]
    
    
    def process_editing(self, input_image: str, scale_factor: float, model_type: str) -> np.ndarray:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the photo.'
            
        if input_image is None:
            #return false_image, false_image, 'Error: fail to load empty file.'
            raise gr.Error('Error: fail to load empty file.')
        frame = cv2.imread(input_image)
        if frame is None:
            #return false_image, false_image, 'Error: fail to load the image.'
            raise gr.Error('Error: fail to load the image.')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            
        if model_type is None or model_type == 'reduce age':
            task_name = 'edit_age'
        else:
            task_name = 'edit_hair'
            
        with torch.no_grad():
            paras = get_video_crop_parameter(frame, self.landmarkpredictor)
            if paras is None:
                #return false_image, false_image, info
                raise gr.Error(self.error_info)
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            x1 = self.transform(frame).unsqueeze(0).to(self.device)
            x2 = align_face(frame, self.landmarkpredictor)
            if x2 is None:
                #return false_image, 'Error: no face detected! Please retry or change the photo.'
                raise gr.Error(self.error_info)
            x2 = self.transform(x2).unsqueeze(dim=0).to(self.device)
            if self.print_log: print('image loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            y_hat = self.pspex(x1=x1, x2=x2, use_skip=self.pspex.opts.use_skip, zero_noise=True, 
                        resize=False, editing_w= - scale_factor* self.editing_w[0:1])
            y_hat = torch.clamp(y_hat, -1, 1)

        return self.tensor2np(y_hat[0])

    def process_vediting(self, input_video: str, scale_factor: float, model_type: str, frame_num: int) -> tuple[list[np.ndarray], str]:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the video.'
            
        if input_video is None:
            #return [false_image], 'default.mp4', 'Error: fail to load empty file.'
            raise gr.Error('Error: fail to load empty file.')
        video_cap = cv2.VideoCapture(input_video)
        success, frame = video_cap.read()
        if success is False:
            #return [false_image], 'default.mp4', 'Error: fail to load the video.' 
            raise gr.Error('Error: fail to load the video.')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)         
        
        if model_type is None or model_type == 'reduce age':
            task_name = 'edit_age'
        else:
            task_name = 'edit_hair'
            
        with torch.no_grad():
            paras = get_video_crop_parameter(frame, self.landmarkpredictor)
            if paras is None:
                #return [false_image], 'default.mp4', info
                raise gr.Error(self.error_info)
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            x1 = self.transform(frame).unsqueeze(0).to(self.device)
            x2 = align_face(frame, self.landmarkpredictor)
            if x2 is None:
                #return [false_image], 'default.mp4', info
                raise gr.Error(self.error_info)
            x2 = self.transform(x2).unsqueeze(dim=0).to(self.device)
            if self.print_log: print('first frame loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter('output.mp4', fourcc, video_cap.get(5), (4*W, 4*H))
            
            viz_frames = []
            for i in range(frame_num):
                if i > 0:
                    success, frame = video_cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
                    x1 = self.transform(frame).unsqueeze(0).to(self.device)
                y_hat = self.pspex(x1=x1, x2=x2, use_skip=self.pspex.opts.use_skip, zero_noise=True, 
                        resize=False, editing_w= - scale_factor * self.editing_w[0:1])
                y_hat = torch.clamp(y_hat, -1, 1)
                videoWriter.write(tensor2cv2(y_hat[0].cpu()))
                if i < min(frame_num, 4):
                    viz_frames += [self.tensor2np(y_hat[0])]
                
            videoWriter.release()    
            
        return viz_frames, 'output.mp4'
    
    
    def process_toonify(self, input_image: str, style_type: str) -> np.ndarray:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the photo.'
            
        if input_image is None:
            raise gr.Error('Error: fail to load empty file.')
            #return false_image, false_image, 'Error: fail to load empty file.'
        frame = cv2.imread(input_image)
        if frame is None:
            raise gr.Error('Error: fail to load the image.')
            #return false_image, false_image, 'Error: fail to load the image.'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            
        if style_type is None or style_type == 'Pixar':
            task_name = 'toonify_pixar'
        elif style_type == 'Cartoon':
            task_name = 'toonify_cartoon'
        else:
            task_name = 'toonify_arcane'
            
        with torch.no_grad():
            paras = get_video_crop_parameter(frame, self.landmarkpredictor)
            if paras is None:
                raise gr.Error(self.error_info)
                #return false_image, false_image, info
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            x1 = self.transform(frame).unsqueeze(0).to(self.device)
            x2 = align_face(frame, self.landmarkpredictor)
            if x2 is None:
                raise gr.Error(self.error_info)
                #return false_image, 'Error: no face detected! Please retry or change the photo.'
            x2 = self.transform(x2).unsqueeze(dim=0).to(self.device)
            if self.print_log: print('image loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            y_hat = self.pspex(x1=x1, x2=x2, use_skip=self.pspex.opts.use_skip, zero_noise=True, resize=False)
            y_hat = torch.clamp(y_hat, -1, 1)

        return self.tensor2np(y_hat[0])    


    def process_vtoonify(self, input_video: str, style_type: str, frame_num: int) -> tuple[list[np.ndarray], str]:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the video.'
            
        if input_video is None:
            raise gr.Error('Error: fail to load empty file.')
            #return [false_image], 'default.mp4', 'Error: fail to load empty file.'
        video_cap = cv2.VideoCapture(input_video)
        success, frame = video_cap.read()
        if success is False:
            raise gr.Error('Error: fail to load the video.')
            #return [false_image], 'default.mp4', 'Error: fail to load the video.'        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)         
        
        if style_type is None or style_type == 'Pixar':
            task_name = 'toonify_pixar'
        elif style_type == 'Cartoon':
            task_name = 'toonify_cartoon'
        else:
            task_name = 'toonify_arcane'
            
        with torch.no_grad():
            paras = get_video_crop_parameter(frame, self.landmarkpredictor)
            if paras is None:
                raise gr.Error(self.error_info)
                #return [false_image], 'default.mp4', info
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            x1 = self.transform(frame).unsqueeze(0).to(self.device)
            x2 = align_face(frame, self.landmarkpredictor)
            if x2 is None:
                raise gr.Error(self.error_info)
                #return [false_image], 'default.mp4', info
            x2 = self.transform(x2).unsqueeze(dim=0).to(self.device)
            if self.print_log: print('first frame loaded')
            self.load_model(task_name)
            if self.print_log: print('model %s loaded'%(task_name))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter('output.mp4', fourcc, video_cap.get(5), (4*W, 4*H))
            
            viz_frames = []
            for i in range(frame_num):
                if i > 0:
                    success, frame = video_cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
                    x1 = self.transform(frame).unsqueeze(0).to(self.device)
                y_hat = self.pspex(x1=x1, x2=x2, use_skip=self.pspex.opts.use_skip, zero_noise=True, resize=False)
                y_hat = torch.clamp(y_hat, -1, 1)
                videoWriter.write(tensor2cv2(y_hat[0].cpu()))
                if i < min(frame_num, 4):
                    viz_frames += [self.tensor2np(y_hat[0])]
                
            videoWriter.release()    
            
        return viz_frames, 'output.mp4'   
    
    
    def process_inversion(self, input_image: str, optimize: str, input_latent: file-object, editing_options: str, 
                          scale_factor: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        #false_image = np.zeros((256,256,3), np.uint8)
        #info = 'Error: no face detected! Please retry or change the photo.'
            
        if input_image is None:
            raise gr.Error('Error: fail to load empty file.')
            #return false_image, false_image, 'Error: fail to load empty file.'
        frame = cv2.imread(input_image)
        if frame is None:
            raise gr.Error('Error: fail to load the image.')
            #return false_image, false_image, 'Error: fail to load the image.'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        task_name = 'inversion'
        self.load_model(task_name)
        if self.print_log: print('model %s loaded'%(task_name))
        if input_latent is not None:
            if '.pt' not in input_latent.name:
                raise gr.Error('Error: the latent format is wrong')
                #return false_image, false_image, 'Error: the latent format is wrong'
            latents = torch.load(input_latent.name)
            if 'wplus' not in latents.keys() or 'f' not in latents.keys():
                raise gr.Error('Error: the latent format is wrong')
                #return false_image, false_image, 'Error: the latent format is wrong'
            wplus = latents['wplus'].to(self.device) # w+
            f = [latents['f'][0].to(self.device)]    # f
        elif optimize == 'Latent optimization':
            wplus, f, _, _, _ = latent_optimization(frame, self.pspex, self.landmarkpredictor, 
                                                                     step=500, device=self.device)
        else:
            with torch.no_grad():
                paras = get_video_crop_parameter(frame, self.landmarkpredictor)
                if paras is None:
                    raise gr.Error(self.error_info)
                    #return false_image, false_image, info
                h,w,top,bottom,left,right,scale = paras
                H, W = int(bottom-top), int(right-left)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
                x1 = self.transform(frame).unsqueeze(0).to(self.device)
                x2 = align_face(frame, self.landmarkpredictor)
                if x2 is None:
                    raise gr.Error(self.error_info)
                    #return false_image, false_image, 'Error: no face detected! Please retry or change the photo.'
                x2 = self.transform(x2).unsqueeze(dim=0).to(self.device)
                if self.print_log: print('image loaded')
                wplus = self.pspex.encoder(x2) + self.pspex.latent_avg.unsqueeze(0)
                _, f =  self.pspex.encoder(x1, return_feat=True)
            
        with torch.no_grad():
            y_hat, _ = self.pspex.decoder([wplus], input_is_latent=True, first_layer_feature=f)
            y_hat = torch.clamp(y_hat, -1, 1)

            if 'Style Mixing' in editing_options:
                torch.manual_seed(seed)
                wplus[:, 8:] = self.pspex.decoder.style(torch.randn(1, 512).to(self.device)).unsqueeze(1).repeat(1,10,1) * 0.7
                y_hat_edit, _ = self.pspex.decoder([wplus], input_is_latent=True, first_layer_feature=f)
            elif 'Attribute Editing' in editing_options: 
                editing_w = self.editing_dicts[editing_options[19:]].to(self.device)
                y_hat_edit, _ = self.pspex.decoder([wplus+scale_factor*editing_w], input_is_latent=True, first_layer_feature=f)
            elif 'Domain Transfer' in editing_options: 
                self.load_G_model(editing_options[17:])
                if self.print_log: print('model %s loaded'%(editing_options[17:]))
                y_hat_edit, _ = self.generator([wplus], input_is_latent=True, first_layer_feature=f)
            else:
                y_hat_edit = y_hat
            y_hat_edit = torch.clamp(y_hat_edit, -1, 1)
            
        return self.tensor2np(y_hat[0]), self.tensor2np(y_hat_edit[0])
