import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BottleNect2D(nn.Module):
    def __init__(self, input_dim, expand = 5):
        super(BottleNect2D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = input_dim, out_channels = expand*input_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(expand*input_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels = expand*input_dim, out_channels = input_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim),
        )
    
    def forward(self, x):
        return x + self.block(x)
    
    
class Discrete_encoder(nn.Module):
    def __init__(self, hidden_dim = 256):
        super(Discrete_encoder, self).__init__()
        
        self.encoder = nn.ModuleList([])
        self.encoder.append(nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=2, padding=0))
        self.encoder.append(nn.BatchNorm2d(16))
        self.encoder.append(BottleNect2D(16))
        self.encoder.append(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride=2, padding=0))
        self.encoder.append(nn.BatchNorm2d(32))
        self.encoder.append(BottleNect2D(32))
        self.encoder.append(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2, padding=0))
        self.encoder.append(nn.BatchNorm2d(64))
        self.encoder.append(BottleNect2D(64))
        self.encoder.append(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=2, padding=0))
        self.encoder.append(nn.BatchNorm2d(128))
        self.encoder.append(BottleNect2D(128))
        self.encoder.append(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=2, padding=0))
        self.encoder.append(nn.BatchNorm2d(256))
        
    def forward(self, x):
        
        for layer in self.encoder:
            x = layer(x)
        
        z = F.adaptive_avg_pool2d(x, 1).unsqueeze(-1)
        
        return z
        
class BottleNect3D(nn.Module):
    def __init__(self, input_dim, expand = 5):
        super(BottleNect3D, self).__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels = input_dim, out_channels = expand*input_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*input_dim),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels = expand*input_dim, out_channels = input_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(input_dim),
        )
    
    def forward(self, x):
        return x + self.block(x)

    
class Discrete_decoder(nn.Module):
    def __init__(self, hidden_dim = 256):
        super(Discrete_decoder, self).__init__()
        
        self.decoder = nn.ModuleList([])
        self.decoder.append(nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride=2, padding=1))
        self.decoder.append(nn.BatchNorm3d(128))
        self.decoder.append(BottleNect3D(128))
        self.decoder.append(nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride=2, padding=1))
        self.decoder.append(nn.BatchNorm3d(64))
        self.decoder.append(BottleNect3D(64))
        self.decoder.append(nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 4, stride=2, padding=1))
        self.decoder.append(nn.BatchNorm3d(32))
        self.decoder.append(BottleNect3D(32))
        self.decoder.append(nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 4, stride=2, padding=1))
        self.decoder.append(nn.BatchNorm3d(16))
        self.decoder.append(BottleNect3D(16))
        self.decoder.append(nn.ConvTranspose3d(in_channels = 16, out_channels = 1, kernel_size = 4, stride=2, padding=1))
        self.decoder.append(BottleNect3D(1))
        
    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        
        return x
        
        
class BottleNect1D(nn.Module):
    def __init__(self, input_dim, expand = 5):
        super(BottleNect1D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, expand*input_dim),
            nn.BatchNorm1d(expand*input_dim),
            nn.ReLU(),
            nn.Linear(expand*input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
    
    def forward(self, x):
        return x + self.block(x)

class BottleNect1D(nn.Module):
    def __init__(self, input_dim, expand = 5):
        super(BottleNect1D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, expand*input_dim),
            nn.BatchNorm1d(expand*input_dim),
            nn.ReLU(),
            nn.Linear(expand*input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
    
    def forward(self, x):
        return x + self.block(x)

class BottleNect1D(nn.Module):
    def __init__(self, input_dim, expand = 5):
        super(BottleNect1D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, expand*input_dim),
            nn.BatchNorm1d(expand*input_dim),
            nn.ReLU(),
            nn.Linear(expand*input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
    
    def forward(self, x):
        return x + self.block(x)

class Continuous(nn.Module):
    def __init__(self, pt_dim = 3, con_dim = 32, latent_dim = 256):
        super(Continuous, self).__init__()
        
        self.de_pt =  nn.Sequential(
            nn.Linear(pt_dim + con_dim + latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            BottleNect1D(latent_dim),
        )
        
        self.de_1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            BottleNect1D(latent_dim),
        )
        
        self.de_2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            BottleNect1D(latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Tanh(),
        )
        
    def forward(self, pt, con, z):
        
        fea = self.de_pt(torch.cat((torch.cat((pt, con), 1), z), 1))
        out = self.de_1(fea) + fea
        out = self.de_2(out)
        
        return out
        
        
class Conditional_UNET_FULL(nn.Module):
    def __init__(self, channel_size = 16, expand = 1):
        super(Conditional_UNET_FULL, self).__init__()
        
        self.channel = channel_size
        
        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 2*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(2*channel_size),
            nn.ReLU(),
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            nn.Conv3d(in_channels = 2*channel_size, out_channels = expand*2*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*2*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*2*channel_size, out_channels = 4*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(4*channel_size),
            nn.ReLU(),
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2, 2),
            nn.Conv3d(in_channels = 4*channel_size, out_channels = expand*4*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*4*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*4*channel_size, out_channels = 8*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(8*channel_size),
            nn.ReLU(),
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool3d(8, 8),
            nn.Conv3d(in_channels = 8*channel_size, out_channels = expand*8*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*8*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*8*channel_size, out_channels = 16*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(16*channel_size),
            nn.ReLU(),
        )
        
        
        self.up1 = nn.Sequential(
            nn.Conv3d(in_channels = (16 + 16)*channel_size, out_channels = expand*16*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*16*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*16*channel_size, out_channels = 16*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(16*channel_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='nearest'),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv3d(in_channels = (16 + 8)*channel_size, out_channels = expand*8*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*8*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*8*channel_size, out_channels = 8*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(8*channel_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv3d(in_channels = (8 + 4)*channel_size, out_channels = expand*4*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*4*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*4*channel_size, out_channels = 4*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(4*channel_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        
        self.up4 = nn.Sequential(
            nn.Conv3d(in_channels = (4 + 2)*channel_size, out_channels = expand*2*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(expand*2*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = expand*2*channel_size, out_channels = 2*channel_size, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(2*channel_size),
            nn.ReLU(),
            nn.Conv3d(in_channels = 2*channel_size, out_channels = 3, kernel_size = 3, stride=1, padding=1),
        )
        
    def forward(self, x, z):
        
        context1 = self.down1(x)
        context2 = self.down2(context1)
        context3 = self.down3(context2)
        context4 = self.down4(context3)
        
        
        out = self.up1(torch.cat((z, context4), 1))
        out = self.up2(torch.cat((context3, out), 1))
        out = self.up3(torch.cat((context2, out), 1))
        out = self.up4(torch.cat((context1, out), 1))
        
        return out
        
        
        
class Conditional_UNET(nn.Module):
    def __init__(self, unet_path):
        super(Conditional_UNET, self).__init__()
        
        unet_full = Conditional_UNET_FULL()
        unet_full.load_state_dict(torch.load(unet_path))
        
        priors = list(unet_full.children())
        
        self.down1 = priors[0]
        self.down2 = priors[1]
        self.down3 = priors[2]
        self.down4 = priors[3]
        self.up1 = priors[4]
        self.up2 = priors[5]
        self.up3 = priors[6]
        
        module_list = []
        
        last = list(priors[-1].children())
    
        for p in range(len(last)-2):
            module_list.append(last[p])
        
        self.up4 = nn.Sequential(*module_list)
        
    def forward(self, x, z):
        
        context1 = self.down1(x)
        context2 = self.down2(context1)
        context3 = self.down3(context2)
        context4 = self.down4(context3)
        
        out = self.up1(torch.cat((z, context4), 1))
        out = self.up2(torch.cat((context3, out), 1))
        out = self.up3(torch.cat((context2, out), 1))
        out = self.up4(torch.cat((context1, out), 1))
        
        return out

class Mapping(nn.Module):
    def __init__(self, latent_dim = 256):
        super(Mapping, self).__init__()
        
        self.mapping = nn.Sequential(
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
            nn.Conv3d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(),
        )
        
    def forward(self, z):
        
        z = self.mapping(z)
        
        return z