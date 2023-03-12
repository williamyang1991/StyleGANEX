import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, max_pooling=False):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.max_pooling = max_pooling
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        # To make E accept more general H*W images, we add global average pooling to 
        # resize all features to 1*1*512 before mapping to latent codes
        if self.max_pooling:
            x = F.adaptive_max_pool2d(x, 1) ##### modified
        else:
            x = F.adaptive_avg_pool2d(x, 1) ##### modified
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(fin, affine=False)
        self.style = nn.Linear(style_dim, fin * 2)

        self.style.bias.data[:fin] = 1
        self.style.bias.data[fin:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out    
    

class FusionLayer(Module):  ##### modified 
    def __init__(self, inchannel, outchannel, use_skip_torgb=True, use_att=0):
        super(FusionLayer, self).__init__()
        
        self.transform = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU())
        self.fusion_out = nn.Conv2d(outchannel*2, outchannel, kernel_size=3, stride=1, padding=1)
        self.fusion_out.weight.data *= 0.01
        self.fusion_out.weight[:,0:outchannel,1,1].data += torch.eye(outchannel) 
        
        self.use_skip_torgb = use_skip_torgb
        if use_skip_torgb:
            self.fusion_skip = nn.Conv2d(3+outchannel, 3, kernel_size=3, stride=1, padding=1)
            self.fusion_skip.weight.data *= 0.01
            self.fusion_skip.weight[:,0:3,1,1].data += torch.eye(3)
            
        self.use_att = use_att
        if use_att:
            modules = []
            modules.append(nn.Linear(512, outchannel))
            for _ in range(use_att):
                modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                modules.append(nn.Linear(outchannel, outchannel))
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.linear = Sequential(*modules)
            self.norm = AdaptiveInstanceNorm(outchannel*2, outchannel)
            self.conv = nn.Conv2d(outchannel*2, 1, 3, 1, 1, bias=True)

    def forward(self, feat, out, skip, editing_w=None):
        x = self.transform(feat)
        # similar to VToonify, use editing vector as condition
        # fuse encoder feature and decoder feature with a predicted attention mask m_E
        # if self.use_att = False, just fuse them with a simple conv layer
        if self.use_att and editing_w is not None:
            label = self.linear(editing_w)
            m_E = (F.relu(self.conv(self.norm(torch.cat([out, abs(out-x)], dim=1), label)))).tanh()
            x = x * m_E
        out = self.fusion_out(torch.cat((out, x), dim=1))
        if self.use_skip_torgb:
            skip = self.fusion_skip(torch.cat((skip, x), dim=1))
        return out, skip

    
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(Conv2d(dim, dim, 3, 1, 1),
                                        nn.LeakyReLU(),
                                        Conv2d(dim, dim, 3, 1, 1))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = x + self.conv_block(x)  
        return self.relu(out)

# trainable light-weight translation network T
# for sketch/mask-to-face translation, 
# we add a trainable T to map y to an intermediate domain where E can more easily extract features. 
class ResnetGenerator(nn.Module):
    def __init__(self, in_channel=19, res_num=2):
        super(ResnetGenerator, self).__init__()
        
        modules = []
        modules.append(Conv2d(in_channel, 16, 3, 2, 1))
        modules.append(nn.LeakyReLU())
        modules.append(Conv2d(16, 16, 3, 2, 1))
        modules.append(nn.LeakyReLU())
        for _ in range(res_num):
            modules.append(ResnetBlock(16))
        for _ in range(2):
            modules.append(nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1))
            modules.append(nn.LeakyReLU())  
        modules.append(Conv2d(16, 64, 3, 1, 1, bias=False))
        modules.append(BatchNorm2d(64))  
        modules.append(PReLU(64))  
        self.model = Sequential(*modules)

    def forward(self, input):
        return self.model(input)
    
class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        
        # for sketch/mask-to-face translation, add a new network T
        if opts.input_nc != 3:
            self.input_label_layer = ResnetGenerator(opts.input_nc, opts.res_num)

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16, 'max_pooling' in opts and opts.max_pooling)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32, 'max_pooling' in opts and opts.max_pooling)
            else:
                style = GradualStyleBlock(512, 512, 64, 'max_pooling' in opts and opts.max_pooling)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        
        # we concatenate pSp features in the middle layers and 
        # add a convolution layer to map the concatenated features to the first-layer input feature f of G.
        self.featlayer = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0) ##### modified
        self.skiplayer = nn.Conv2d(768, 3, kernel_size=1, stride=1, padding=0) ##### modified
        
        # skip connection
        if 'use_skip' in opts and opts.use_skip: ##### modified
            self.fusion = nn.ModuleList()
            channels = [[256,512], [256,512], [256,512], [256,512], [128,512], [64,256], [64,128]]
            # opts.skip_max_layer: how many layers are skipped to the decoder
            for inc, outc in channels[:max(1, min(7, opts.skip_max_layer))]: # from 4 to 256
                self.fusion.append(FusionLayer(inc, outc, opts.use_skip_torgb, opts.use_att))

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    # return_feat: return f
    # return_full: return f and the skipped encoder features
    # return [out, feats]
    # out is the style latent code w+
    # feats[0] is f for the 1st conv layer, feats[1] is f for the 1st torgb layer
    # feats[2-8] is the skipped encoder features 
    def forward(self, x, return_feat=False, return_full=False): ##### modified      
        if x.shape[1] != 3:
            x = self.input_label_layer(x)
        else:
            x = self.input_layer(x)
        c256 = x ##### modified
        
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:    ##### modified
                c128 = x
            elif i == 6:
                c1 = x
            elif i == 10: ##### modified
                c21 = x   ##### modified
            elif i == 15: ##### modified
                c22 = x   ##### modified
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        
        if not return_feat:
            return out
        
        feats = [self.featlayer(torch.cat((c21, c22, c2), dim=1)), self.skiplayer(torch.cat((c21, c22, c2), dim=1))]
        
        if return_full: ##### modified
            feats += [c2, c2, c22, c21, c1, c128, c256]
            
        return out, feats

    
    # only compute the first-layer feature f
    # E_F in the paper
    def get_feat(self, x): ##### modified     
        # for sketch/mask-to-face translation
        # use a trainable light-weight translation network T
        if x.shape[1] != 3:
            x = self.input_label_layer(x)
        else:
            x = self.input_layer(x)
        
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 10: ##### modified
                c21 = x   ##### modified
            elif i == 15: ##### modified
                c22 = x   ##### modified
            elif i == 20:
                c2 = x
                break
        return self.featlayer(torch.cat((c21, c22, c2), dim=1))
    
class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
