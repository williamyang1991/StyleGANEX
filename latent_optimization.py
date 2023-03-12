import models.stylegan2.lpips as lpips
from torch import autograd, optim
from torchvision import transforms, utils
from tqdm import tqdm
import torch
from scripts.align_all_parallel import align_face
from utils.inference_utils import noise_regularize, noise_normalize_, get_lr, latent_noise, visualize

def latent_optimization(frame, pspex, landmarkpredictor, step=500, device='cuda'):   
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
    with torch.no_grad():

        noise_sample = torch.randn(1000, 512, device=device)
        latent_out = pspex.decoder.style(noise_sample)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / 1000) ** 0.5              
        
        y = transform(frame).unsqueeze(dim=0).to(device)
        I_ = align_face(frame, landmarkpredictor)
        I_ = transform(I_).unsqueeze(dim=0).to(device)
        wplus = pspex.encoder(I_) + pspex.latent_avg.unsqueeze(0)
        _, f = pspex.encoder(y, return_feat=True)
        latent_in = wplus.detach().clone()
        feat = [f[0].detach().clone(), f[1].detach().clone()]
        
        
    
    # wplus and f to optimize
    latent_in.requires_grad = True
    feat[0].requires_grad = True
    feat[1].requires_grad = True
        
    noises_single = pspex.decoder.make_noise()
    basic_height, basic_width = int(y.shape[2]*32/256), int(y.shape[3]*32/256)
    noises = []
    for noise in noises_single:
        noises.append(noise.new_empty(y.shape[0], 1, max(basic_height, int(y.shape[2]*noise.shape[2]/256)), 
                                      max(basic_width, int(y.shape[3]*noise.shape[2]/256))).normal_())
    for noise in noises:
        noise.requires_grad = True

    init_lr=0.05
    optimizer = optim.Adam(feat + noises, lr=init_lr)
    optimizer2 = optim.Adam([latent_in], lr=init_lr)
    noise_weight = 0.05 * 0.2
    
    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:
        t = i / step
        lr = get_lr(t, init_lr)
        optimizer.param_groups[0]["lr"] = lr
        optimizer2.param_groups[0]["lr"] = get_lr(t, init_lr)

        noise_strength = latent_std * noise_weight * max(0, 1 - t / 0.75) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        y_hat, _ = pspex.decoder([latent_n], input_is_latent=True, randomize_noise=False, 
                                 first_layer_feature=feat, noise=noises) 


        batch, channel, height, width = y_hat.shape

        if height > y.shape[2]:
            factor = height // y.shape[2]

            y_hat = y_hat.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            y_hat = y_hat.mean([3, 5])

        p_loss = percept(y_hat, y).sum()
        n_loss = noise_regularize(noises) * 1e3

        loss = p_loss + n_loss

        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()

        noise_normalize_(noises)

        ''' for visualization
        if (i + 1) % 100 == 0 or i == 0:
            viz = torch.cat((y_hat,y,y_hat-y), dim=3)
            visualize(torch.clamp(viz[0].cpu(),-1,1), 60)  
        '''
        
        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" lr: {lr:.4f}"
            )
        )    
    
    return latent_n, feat, noises, wplus, f