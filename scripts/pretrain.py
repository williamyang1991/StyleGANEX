import os
import sys
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
from torch import autograd, optim
from utils.inference_utils import save_image
from models.psp import pSp
from models.stylegan2.model import Downsample
import models.stylegan2.lpips as lpips

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
      
class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="StyleGANEX Pretraining")
        self.parser.add_argument('--exp_dir', type=str, default='./logs/styleganex_pretrain/', help='Path to experiment output directory')
        self.parser.add_argument("--ckpt", type=str, default='./pretrained_models/psp_ffhq_encode.pt', help="path of the original psp model")
        self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--max_steps', default=5000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt    
    
if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print('*'*98)
    
    device = "cuda"
    
    os.makedirs(args.exp_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.exp_dir, 'checkpoints')
    log_path = os.path.join(args.exp_dir, 'logs')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.ckpt
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    if 'toonify_weights' not in opts:
        opts['toonify_weights'] = None
    opts = Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)
    requires_grad(pspex, False)
    requires_grad(pspex.encoder.featlayer, True)
    requires_grad(pspex.encoder.skiplayer, True)
    
    down = Downsample([1, 3, 3, 1], 2).to(device)
    requires_grad(down, False)
    
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
    requires_grad(percept.model.net, False)
    
    e_optim = optim.Adam(
        list(pspex.encoder.featlayer.parameters()) + list(pspex.encoder.skiplayer.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
    )
    
    pbar = tqdm(range(args.max_steps), initial=0, dynamic_ncols=True, smoothing=0.01)
    recon_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        with torch.no_grad():
            noise_sample = torch.randn(args.batch_size, 512).cuda()
            img_gen, _ = pspex.decoder([noise_sample], input_is_latent=False, truncation=0.7, truncation_latent=0, randomize_noise=False)
            img_gen = torch.clamp(img_gen, -1, 1).detach()        
            img_real = img_gen.clone()
            real_input = down(down(img_gen)).detach()
            style = pspex.encoder(real_input) + pspex.latent_avg.unsqueeze(0)
            if idx == 0:
                samplein = real_input.clone().detach()
                samplestyle = style.clone().detach()

        _, feat = pspex.encoder(real_input, return_feat=True)
        fake_img, _ = pspex.decoder([style], input_is_latent=True, randomize_noise=False, first_layer_feature=feat)

        recon_loss = F.mse_loss(fake_img, img_real) * 10
        perct_loss = percept(down(fake_img), down(img_real).detach()).sum()
        e_loss = recon_loss + perct_loss

        loss_dict["er"] = recon_loss
        loss_dict["ef"] = perct_loss        

        pspex.zero_grad()
        e_loss.backward()  
        e_optim.step()

        er_loss_val = loss_dict["er"].mean().item()
        ef_loss_val = loss_dict["ef"].mean().item()

        pbar.set_description(
            (
                f"iter: {idx:d}; er: {er_loss_val:.3f}; ef: {ef_loss_val:.3f}"
            )
        )

        if idx % args.image_interval == 0 or (idx+1) == args.max_steps:
            with torch.no_grad():
                _, sample_feat = pspex.encoder(samplein, return_feat=True)
                sample, _ = pspex.decoder([samplestyle], input_is_latent=True, randomize_noise=False, first_layer_feature=sample_feat)
                sample = torch.cat((samplein, down(down(sample))), dim=0)
                save_image(torchvision.utils.make_grid(sample, args.batch_size, 1), f"%s/%05d.jpg"%(log_path, idx+1))    
                
    save_dict = {
        'state_dict': pspex.state_dict(),
        'opts': vars(pspex.opts)
    }
    if pspex.opts.start_from_latent_avg:
        save_dict['latent_avg'] = pspex.latent_avg
    torch.save(
        save_dict,
        f"%s/%05d.pt"%(checkpoint_path, idx+1),
    )