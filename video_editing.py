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

        self.parser = argparse.ArgumentParser(description="StyleGANEX Video Editing")
        self.parser.add_argument("--data_path", type=str, default='./data/390.mp4', help="path of the target image/video")
        self.parser.add_argument("--ckpt", type=str, default='pretrained_models/styleganex_toonify_cartoon.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output results")
        self.parser.add_argument("--scale_factor", type=float, default=1.0, help="scale of the editing degree")
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
    editing_w = None
    if 'editing_w' in ckpt.keys():
        editing_w = ckpt['editing_w'].clone().to(device)[0:1] * args.scale_factor

    modelname = 'pretrained_models/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    print('Load models successfully!')

    video_path = args.data_path
    video_cap = cv2.VideoCapture(video_path)
    success, frame = video_cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    paras = get_video_crop_parameter(frame, landmarkpredictor)
    assert paras is not None, 'StyleGANEX uses dlib.get_frontal_face_detector but sometimes it fails to detect a face. \
                               You can try several times or use other videos until a face is detected, \
                               then switch back to the original video.'
    h,w,top,bottom,left,right,scale = paras
    H, W = int(bottom-top), int(right-left)
    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

    x1 = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        x2 = align_face(frame, landmarkpredictor)
        x2 = transform(x2).unsqueeze(dim=0).to(device)

    save_name = '%s/%s_%s'%(args.output_path, os.path.basename(video_path).split('.')[0],  os.path.basename(args.ckpt).split('.')[0])

    num = int(video_cap.get(7))

    if num == 1: # input is image
        save_name = save_name + '.jpg'
    else:        # input is video
        save_name = save_name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(save_name, fourcc, video_cap.get(5), (4*W, 4*H))    

    with torch.no_grad():
        for i in tqdm(range(num)):
            if i > 0:
                success, frame = video_cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

            x1 = transform(frame).unsqueeze(0).to(device)
            y_hat = pspex(x1=x1, x2=x2, use_skip=pspex.opts.use_skip, zero_noise=True, 
                           resize=False, editing_w=editing_w)
            y_hat = torch.clamp(y_hat, -1, 1)

            if num > 1:
                videoWriter.write(tensor2cv2(y_hat[0].cpu()))

    if num == 1:
        save_image(y_hat[0].cpu(), save_name)
        print('Image editing successfully!')
    else:
        videoWriter.release()
        print('Video editing successfully!')

