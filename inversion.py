import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from models.psp import pSp
import torch
import dlib
import cv2
import PIL
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from argparse import Namespace
from datasets import augmentations
from scripts.align_all_parallel import align_face
from latent_optimization import latent_optimization
from utils.inference_utils import save_image, load_image, visualize, get_video_crop_parameter, tensor2cv2, tensor2label, labelcolormap

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="StyleGANEX Inversion")
        self.parser.add_argument("--data_path", type=str, default='./data/ILip77SbmOE.png', help="path of the target image")
        self.parser.add_argument("--ckpt", type=str, default='pretrained_models/styleganex_inversion.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    

if __name__ == "__main__":
    
    parser = TestOptions()
    args = parser.parse()
    print('*'*98)

    device = "cpu" if args.cpu else "cuda"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])

    ckpt = torch.load(args.ckpt, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.ckpt
    opts['device'] =  device
    opts = Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)

    modelname = 'pretrained_models/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    print('Load models successfully!')

    image_path = args.data_path
    with torch.no_grad():
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        paras = get_video_crop_parameter(frame, landmarkpredictor)
        assert paras is not None, 'StyleGANEX uses dlib.get_frontal_face_detector but sometimes it fails to detect a face. \
                               You can try several times or use other videos until a face is detected, \
                               then switch back to the original video.'
        h,w,top,bottom,left,right,scale = paras
        H, W = int(bottom-top), int(right-left)
        frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

    wplus_hat, f_hat, noises_hat, _, _ = latent_optimization(frame, pspex, landmarkpredictor, step=500, device=device)

    with torch.no_grad():
        y_hat, _ = pspex.decoder([wplus_hat], input_is_latent=True, randomize_noise=False, 
                                     first_layer_feature=f_hat, noise=noises_hat)

        y_hat = torch.clamp(y_hat, -1, 1)

    save_dict = {
        'wplus': wplus_hat.detach().cpu(),
        'f': [f.detach().cpu() for f in f_hat],
        #'noise': [n.detach().cpu() for n in noises_hat],
    }    
    torch.save(save_dict, '%s/%s_inversion.pt'%(args.output_path, os.path.basename(image_path).split('.')[0]))
    save_image(y_hat[0].cpu(), '%s/%s_inversion.jpg'%(args.output_path, os.path.basename(image_path).split('.')[0]))

    # how to use the saved pt
    '''
    latents = torch.load('./output/XXXXX_inversion.pt')
    wplus_hat = latents['wplus'].to(device)
    f_hat = [latents['f'][0].to(device)]
    with torch.no_grad():
        y_hat, _ = pspex.decoder([wplus_hat], input_is_latent=True, randomize_noise=True, 
                                     first_layer_feature=f_hat, noise=None)  
        y_hat = torch.clamp(y_hat, -1, 1)
    '''

    print('Inversion successfully!')
