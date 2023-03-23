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

        self.parser = argparse.ArgumentParser(description="StyleGANEX Image Translation")
        self.parser.add_argument("--data_path", type=str, default='./data/ILip77SbmOE.png', help="path of the target image")
        self.parser.add_argument("--ckpt", type=str, default='pretrained_models/styleganex_sr32.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--use_raw_data", action="store_true", help="if true, input image needs no pre-procssing")
        self.parser.add_argument("--resize_factor", type=int, default=32, help="super resolution resize factor")
        self.parser.add_argument("--number", type=int, default=4, help="output number of multi-modal translation")
        self.parser.add_argument("--parsing_model_ckpt", type=str, default='pretrained_models/faceparsing.pth', help="path of the parsing model")
        
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
    

    image_path = args.data_path
    save_name = '%s/%s_%s'%(args.output_path, os.path.basename(image_path).split('.')[0],  os.path.basename(args.ckpt).split('.')[0])

    modelname = 'pretrained_models/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    if opts.dataset_type == 'ffhq_seg_to_face' and not args.use_raw_data:
        from models.bisenet.model import BiSeNet
        maskpredictor = BiSeNet(n_classes=19)
        maskpredictor.load_state_dict(torch.load(args.parsing_model_ckpt))
        maskpredictor.to(device).eval()
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if opts.dataset_type == 'ffhq_super_resolution':
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.use_raw_data:
            x, y = frame.shape[0:2]
            tmp = PIL.Image.fromarray(np.uint8(frame)).resize((int(y) * args.resize_factor // 4, int(x) * args.resize_factor // 4))
            frame = np.array(tmp)
        paras = get_video_crop_parameter(frame, landmarkpredictor)
        assert paras is not None, 'StyleGANEX uses dlib.get_frontal_face_detector but sometimes it fails to detect a face. \
                               You can try several times or use other videos until a face is detected, \
                               then switch back to the original video.'
        h,w,top,bottom,left,right,scale = paras
        H, W = int(bottom-top), int(right-left)
        frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
        if not args.use_raw_data:
            x1 = PIL.Image.fromarray(np.uint8(frame))
            x1 = augmentations.BilinearResize(factors=[args.resize_factor // 4])(x1)
            x1.save(save_name + '_input.png')
            x1_up = x1.resize((W, H))
            x2_up = align_face(np.array(x1_up), landmarkpredictor)
            x1_up = transforms.ToTensor()(x1_up).unsqueeze(dim=0).to(device) * 2 - 1
        else:
            x1_up = transform(frame).unsqueeze(0).to(device)   
            x2_up = align_face(frame, landmarkpredictor)
        x2_up = transform(x2_up).unsqueeze(dim=0).to(device)
        x1 = x1_up
        x2 = x2_up
    elif opts.dataset_type == 'ffhq_sketch_to_face':
        # no pre-processing supported, only accept one-channel sketch image
        x1 = transforms.ToTensor()(PIL.Image.open(image_path)).unsqueeze(0).to(device)
        x2 = None
    elif opts.dataset_type == 'ffhq_seg_to_face':
        if not args.use_raw_data:
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            paras = get_video_crop_parameter(frame, landmarkpredictor)        
            assert paras is not None, 'StyleGANEX uses dlib.get_frontal_face_detector but sometimes it fails to detect a face. \
                               You can try several times or use other videos until a face is detected, \
                               then switch back to the original video.'
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]        
            # convert face image to segmentation mask
            x1 = to_tensor(frame).unsqueeze(0).to(device)
            # upsample image for precise segmentation
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear') 
            x1 = maskpredictor(x1)[0]
            x1 = F.interpolate(x1, scale_factor=0.5).argmax(dim=1)
            cv2.imwrite(save_name+'_input.png', x1.squeeze(0).cpu().numpy())
            x1 = F.one_hot(x1, num_classes=19).permute(0, 3, 1, 2).float().to(device)
        else:
            x1 = PIL.Image.open(image_path)
            x1 = augmentations.ToOneHot(opts.label_nc)(x1)
            x1 = transforms.ToTensor()(x1).unsqueeze(dim=0).float().to(device)
        x1_viz = transform(tensor2label(x1[0], 19)/192)
        save_image(x1_viz, save_name+'_input_viz.jpg')
        x2 = None    
    else:
        assert False, 'The input model %s does not support image translation task'%(args.ckpt)

    print('Load models successfully!')
    
    with torch.no_grad():
        if opts.dataset_type == 'ffhq_super_resolution':
            y_hat = torch.clamp(pspex(x1=x1, x2=x2, use_skip=pspex.opts.use_skip, resize=False), -1, 1)
            save_image(y_hat[0].cpu(), save_name+'.jpg')
        else:
            pspex.train()
            for i in range(args.number):
                y_hat = pspex(x1=x1, x2=x2, resize=False, latent_mask=[8,9,10,11,12,13,14,15,16,17], use_skip=pspex.opts.use_skip,
                                  inject_latent = pspex.decoder.style(torch.randn(1, 512).to(device)).unsqueeze(1).repeat(1,18,1) * 0.7)  
                y_hat = torch.clamp(y_hat, -1, 1)
                save_image(y_hat[0].cpu(), save_name+'_%d.jpg'%(i))
            pspex.eval()

    print('Image translation successfully!')
