"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import torch.nn.functional as F

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts, ckpt=None):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights(ckpt)

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self, ckpt=None):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            if ckpt is None:
                ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)
        # for video toonification, we load G0' model
        if self.opts.toonify_weights is not None: ##### modified
            ckpt = torch.load(self.opts.toonify_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.opts.toonify_weights = None

    # x1: image for first-layer feature f. 
    # x2: image for style latent code w+. If not specified, x2=x1.
    # inject_latent: for sketch/mask-to-face translation, another latent code to fuse with w+
    # latent_mask: fuse w+ and inject_latent with the mask (1~7 use w+ and 8~18 use inject_latent)
    # use_feature: use f. Otherwise, use the orginal StyleGAN first-layer constant 4*4 feature 
    # first_layer_feature_ind: always=0, means the 1st layer of G accept f
    # use_skip: use skip connection.
    # zero_noise: use zero noises. 
    # editing_w: the editing vector v for video face editing
    def forward(self, x1, x2=None, resize=True, latent_mask=None, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, use_feature=True, 
                first_layer_feature_ind=0, use_skip=False, zero_noise=False, editing_w=None): ##### modified
        
        feats = None # f and the skipped encoder features
        codes, feats = self.encoder(x1, return_feat=True, return_full=use_skip) ##### modified
        if x2 is not None: ##### modified
            codes = self.encoder(x2) ##### modified
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        # E_W^{1:7}(T(x1)) concatenate E_W^{8:18}(w~)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0
                    
        first_layer_feats, skip_layer_feats, fusion = None, None, None ##### modified            
        if use_feature: ##### modified
            first_layer_feats = feats[0:2] # use f
            if use_skip: ##### modified
                skip_layer_feats = feats[2:] # use skipped encoder feature
                fusion = self.encoder.fusion # use fusion layer to fuse encoder feature and decoder feature.
            
        images, result_latent = self.decoder([codes],
                                             input_is_latent=True,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents,
                                             first_layer_feature=first_layer_feats,
                                             first_layer_feature_ind=first_layer_feature_ind,
                                             skip_layer_feature=skip_layer_feats,
                                             fusion_block=fusion,
                                             zero_noise=zero_noise,
                                             editing_w=editing_w) ##### modified

        if resize:
            if self.opts.output_size == 1024:  ##### modified
                images = F.adaptive_avg_pool2d(images, (images.shape[2]//4, images.shape[3]//4))  ##### modified
            else:
                images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
