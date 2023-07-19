import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd  ##### modified
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from models.stylegan2.model import Discriminator ##### modified
from training.ranger import Ranger
from models.stylegan2.simple_augment import random_apply_affine
from datasets.ffhq_degradation_dataset import FFHQDegradationDataset ##### modified, for blind SR

class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize network
        self.net = pSp(self.opts).to(self.device)
        if self.opts.adv_lambda > 0:  ##### modified, add discriminator
            self.discriminator = Discriminator(1024, channel_multiplier=2, img_channel=3)
            if self.opts.checkpoint_path is not None:
                ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
                if 'discriminator' in ckpt.keys():
                    print('Loading discriminator from checkpoint: {}'.format(self.opts.checkpoint_path))
                    self.discriminator.load_state_dict(ckpt['discriminator'], strict=False)
            self.discriminator = self.discriminator.to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=self.opts.learning_rate)
            
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
            
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        # for sketch/mask-to-face translation, indicate which layers from x, which layers from y
        if self.opts.use_latent_mask:  ##### modified
            self.latent_mask = [int(l) for l in self.opts.latent_mask.split(",")]
        
        # for video face editing, the editing vector v
        self.editing_w = None
        if self.opts.editing_w_path is not None:
            self.editing_w = torch.load(self.opts.editing_w_path).to(self.device)
            
        # for video face editing, to augment face attribute when generating training data
        self.directions = None
        if self.opts.direction_path is not None:
            self.directions = torch.load(self.opts.direction_path).to(self.device) 
        
    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                
                #************************ Data Preparation **************************
                
                # x is the input, y is the ground truth
                # the faces in x and y are aligned, we will apply geometric transformation to make them unaligned.
                x, y = batch 
                
                # for video face editing, generating paired data (x,y)
                editing_w = None
                if self.opts.generate_training_data and self.editing_w is not None:
                    with torch.no_grad():
                        noise_sample = torch.randn(x.shape[0], 512).to(self.device)
                        ws = self.net.decoder.style(noise_sample).unsqueeze(1).repeat(1,18,1)
                        ws = ws + self.directions[torch.randint(0, self.directions.shape[0], (x.shape[0],))]
                        x, _ = self.net.decoder([ws], input_is_latent=True, truncation=0.75, truncation_latent=0, randomize_noise=False)
                        x = F.adaptive_avg_pool2d(x, (x.shape[2]//4, x.shape[3]//4)).detach()
                        scale_factor = np.random.choice([0.0,0.25,0.5,0.75,1.0,1.25], 1)[0]
                        editing_w = self.editing_w[torch.randint(0, self.editing_w.shape[0], (1,))] * scale_factor
                        y, _ = self.net.decoder([ws], input_is_latent=True, truncation=0.75, truncation_latent=0, 
                                                randomize_noise=False, editing_w=editing_w)
                        y = y.detach()
                
                x, y = x.to(self.device).float(), y.to(self.device).float()
                
                # the shape of y should be H*W or H/8*W/8, the shape of x should always be H/8*W/8
                scale = int(y.shape[2] // x.shape[2])
                assert(int(y.shape[3] // x.shape[3]) == scale)
                
                # prepare aligned images for w+ extraction
                x_tilde = None
                y_tilde = y.clone() if scale ==1 else F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
                # crop the centered 256*256 region from a H/8*W/8 image 
                if self.opts.crop_face: 
                    crop_size = int((x.shape[2] - 256) // 2) 
                    x_tilde = x.clone()
                    if crop_size > 0:
                        x_tilde = x_tilde[:,:,crop_size:-crop_size,crop_size:-crop_size]
                        if self.opts.use_latent_mask:
                            y_tilde = y_tilde[:,:,crop_size:-crop_size,crop_size:-crop_size]
                
                # for flicker suppression loss in video-related tasks
                y0_hat = None
                if self.opts.tmp_lambda > 0 and self.global_step * 2 >= self.opts.max_steps: 
                    if self.opts.use_latent_mask: # for sketch/mask-to-face translation. not used in the paper
                        y0_hat = self.net.forward(x1=x, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     latent_mask=self.latent_mask, inject_latent=self.net.encoder(y_tilde), 
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w)
                    else:
                        y0_hat = self.net.forward(x1=x, x2=x_tilde, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w)
                
                # making the faces unaligned for the following training
                if self.opts.affine_augment:  
                    x, affine_T = random_apply_affine(x, 0.2, None)
                    y, _ = random_apply_affine(y, 1.0, affine_T)
                    x = x.detach()
                    y = y.detach()
                    if y0_hat is not None and self.opts.tmp_lambda > 0: 
                        y0_hat, _ = random_apply_affine(y0_hat, 1.0, affine_T)
                
                # making the resolution of the image variable
                if self.opts.random_crop:  
                    _, _, h, w = x.shape
                    th, tw = torch.randint(32, 41, size=(1,)).item() * 8, torch.randint(32, 41, size=(1,)).item() * 8
                    i, j = torch.randint(0, h - th + 1, size=(1,)).item(), torch.randint(0, w - tw + 1, size=(1,)).item()
                    x = x[:,:,i:i+th,j:j+tw].detach()
                    y = y[:,:,i*scale:(i+th)*scale,j*scale:(j+tw)*scale].detach()
                    if y0_hat is not None and self.opts.tmp_lambda > 0: 
                        y0_hat = y0_hat[:,:,i*scale:(i+th)*scale,j*scale:(j+tw)*scale]
                
                # if opts.crop_face=False, using unaligned faces to extract w+
                if self.opts.use_latent_mask and (not self.opts.crop_face): 
                    y_tilde = y.clone() if scale == 1 else F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
                    
                # Now we have prepare the input data (x,x_tilde,y_tilde) and ground truth data (y, y0_hat)
                # x is the input image, y is the ground truth output
                # (if opts.crop_face=True) x_tilde is the cropped aligned version of x, y_tilde is the cropped aligned version of y 
                # y0_hat is the geometrically transformed output, which is used to suppress flickers
                
                #************************ generate y_hat given the data ************************** 
                
                # y_hat is the output image, latent is the extracted w+
                if self.opts.use_latent_mask: # for sketch/mask-to-face translation
                    y_hat, latent = self.net.forward(x1=x, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise, 
                                                     latent_mask=self.latent_mask,  inject_latent=self.net.encoder(y_tilde),
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w, return_latents=True)  
                else: # for other tasks
                    y_hat, latent = self.net.forward(x1=x, x2=x_tilde, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w, return_latents=True)  
                # adversarial loss
                if self.opts.adv_lambda > 0: 
                    d_loss_dict = self.train_discriminator(y, y_hat)
                
                # calculate losses
                loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, y0_hat)
                
                if self.opts.adv_lambda > 0:
                    loss_dict = {**d_loss_dict, **loss_dict}
                
                loss.backward()
                self.optimizer.step()

                #************************ logging and saving model************************** 
                
                # Logging related
                with torch.no_grad(): ##### modified for SR task, since x, y and y_hat may have different resolution
                    y = F.adaptive_avg_pool2d(y, (x.shape[2], x.shape[3]))
                    y_hat = F.adaptive_avg_pool2d(y_hat, (x.shape[2], x.shape[3]))
                    x = torch.clamp(x, -1, 1)  
                    
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0:
                    self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch

            editing_w = None
            if self.editing_w is not None:
                editing_w = self.editing_w[torch.randint(0, self.editing_w.shape[0], (1,))]

            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                scale = int(y.shape[2] // x.shape[2])
                assert(int(y.shape[3] // x.shape[3]) == scale)
                
                # prepare aligned images for w+ extraction
                x_tilde = None
                y_tilde = y.clone() if scale ==1 else F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
                # crop the centered 256*256 region from a H/8*W/8 image 
                if self.opts.crop_face:  
                    crop_size = int((x.shape[2] - 256) // 2) 
                    x_tilde = x.clone()
                    if crop_size > 0:
                        x_tilde = x_tilde[:,:,crop_size:-crop_size,crop_size:-crop_size]
                        if self.opts.use_latent_mask:
                            y_tilde = y_tilde[:,:,crop_size:-crop_size,crop_size:-crop_size]
                            
                # for flicker suppression loss in video-related tasks
                y0_hat = None
                if self.opts.tmp_lambda > 0 and self.global_step * 2 >= self.opts.max_steps: 
                    if self.opts.use_latent_mask: # for sketch/mask-to-face translation. not used in the paper
                        y0_hat = self.net.forward(x1=x, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     latent_mask=self.latent_mask, inject_latent=self.net.encoder(y_tilde), 
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w)
                    else:
                        y0_hat = self.net.forward(x1=x, x2=x_tilde, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w)   
                    y0_hat = y0_hat.detach()              

                # making the faces unaligned for the following training
                if self.opts.affine_augment:  
                    x, affine_T = random_apply_affine(x, 0.2, None)
                    y, _ = random_apply_affine(y, 1.0, affine_T)
                    x = x.detach()
                    y = y.detach()
                    if y0_hat is not None and self.opts.tmp_lambda > 0: 
                        y0_hat, _ = random_apply_affine(y0_hat, 1.0, affine_T)
                        y0_hat = y0_hat.detach()
                
                # making the resolution of the image variable
                if self.opts.random_crop:  
                    _, _, h, w = x.shape
                    th, tw = torch.randint(32, 41, size=(1,)).item() * 8, torch.randint(32, 41, size=(1,)).item() * 8
                    i, j = torch.randint(0, h - th + 1, size=(1,)).item(), torch.randint(0, w - tw + 1, size=(1,)).item()
                    x = x[:,:,i:i+th,j:j+tw].detach()
                    y = y[:,:,i*scale:(i+th)*scale,j*scale:(j+tw)*scale].detach()
                    if y0_hat is not None and self.opts.tmp_lambda > 0: 
                        y0_hat = y0_hat[:,:,i*scale:(i+th)*scale,j*scale:(j+tw)*scale].detach()                   
          

                # if opts.crop_face=False, using unaligned faces to extract w+
                if self.opts.use_latent_mask and (not self.opts.crop_face): 
                    y_tilde = y.clone() if scale == 1 else F.interpolate(y, (x.shape[2], x.shape[3]), mode='bilinear')
                
                # y_hat is the output image, latent is the extracted w+
                if self.opts.use_latent_mask: # for sketch/mask-to-face translation
                    y_hat, latent = self.net.forward(x1=x, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise, 
                                                     latent_mask=self.latent_mask,  inject_latent=self.net.encoder(y_tilde),
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w, return_latents=True)  
                else: # for other tasks
                    y_hat, latent = self.net.forward(x1=x, x2=x_tilde, resize=(x.shape[2:]==y.shape[2:]), zero_noise=self.opts.zero_noise,
                                                     first_layer_feature_ind=self.opts.feat_ind, use_skip=self.opts.use_skip,
                                                     editing_w=editing_w, return_latents=True)                  
                
                # adversarial loss             
                if self.opts.adv_lambda > 0: 
                    cur_d_loss_dict = self.validate_discriminator(y, y_hat)
                
                loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, y0_hat) 
                
                if self.opts.adv_lambda > 0: 
                    cur_loss_dict = {**cur_d_loss_dict, **cur_loss_dict}
                
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            with torch.no_grad(): ##### modified for SR task
                y = F.adaptive_avg_pool2d(y, (x.shape[2], x.shape[3]))
                y_hat = F.adaptive_avg_pool2d(y_hat, (x.shape[2], x.shape[3]))
                x = torch.clamp(x, -1, 1)  ##### modified
            
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            # Log images of first batch to wandb
            if self.opts.use_wandb and batch_idx == 0:
                self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        if hasattr(self.opts, 'pretrain_model') and self.opts.pretrain_model == 'input_label_layer': ##### modified
            params = list(self.net.encoder.input_label_layer.parameters())
        else:
            params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        if self.opts.blind_sr:
            import yaml
            with open("./configs/dataset_config.yml", 'r') as stream:
                parsed_yaml=yaml.safe_load(stream)
            parsed_yaml['datasets']['train']['dataroot_gt'] = dataset_args['train_target_root']
            factors = [int(f) for f in self.opts.resize_factors.split(",")]
            if '320' in self.opts.dataset_type:
                parsed_yaml['datasets']['train']['scale'] = 1
            rescale = parsed_yaml['datasets']['train']['scale']
            parsed_yaml['datasets']['train']['downsample_range'] = [min(factors) * 0.75 * rescale, max(factors)* 1.5 * rescale]
            train_dataset = FFHQDegradationDataset(parsed_yaml['datasets']['train'])
        else:
            train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, y0_hat=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0: 
            loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda
        if self.opts.adv_lambda > 0:  ##### modified
            loss_g = F.softplus(-self.discriminator(y_hat)).mean()
            loss_dict['loss_g'] = float(loss_g)
            loss += loss_g * self.opts.adv_lambda
        if self.opts.tmp_lambda > 0 and y0_hat is not None:  ##### modified
            loss_tmp = ((y_hat-y0_hat)**2).mean()
            loss_dict['loss_tmp'] = float(loss_tmp)
            loss += loss_tmp * self.opts.tmp_lambda * min(1, 4.0*(self.global_step/self.opts.max_steps-0.5))
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        if self.opts.adv_lambda > 0:  ##### modified
            save_dict['discriminator'] = self.discriminator.state_dict()
        if self.opts.editing_w_path is not None:
            save_dict['editing_w'] = self.editing_w.cpu()
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
    
    ##### modified
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['loss_d_real'] = float(real_loss)
        loss_dict['loss_d_fake'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, real_img, fake_img):
        loss_dict = {}
        self.requires_grad(self.discriminator, True)

        real_pred = self.discriminator(real_img)
        fake_pred = self.discriminator(fake_img.detach())
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['loss_d'] = float(loss)
        loss = loss * self.opts.adv_lambda 

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_img = real_img.detach()
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = self.discriminator_r1_loss(real_pred, real_img)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['loss_r1'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict
    
    def validate_discriminator(self, real_img, fake_img):
        with torch.no_grad():
            loss_dict = {}
            real_pred = self.discriminator(real_img)
            fake_pred = self.discriminator(fake_img.detach())
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['loss_d'] = float(loss)
            loss = loss * self.opts.adv_lambda 
            return loss_dict