import os
from functools import partial

import torch
import torch.nn as nn 

import numpy as np


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(           #Remplacer par fichier config
        self,
        image_size = (224,224),
        channel_y = 3,
        channel_x = 1,
        timesteps = 2000
        ):
        
        super().__init__()
        
        self.image_size = image_size
        self.channel_y = channel_y
        self.channel_x = channel_x
        self.timesteps = timesteps
    
        alphas = np.linspace(1e-6,0.01,timesteps)
        gammas = np.cumprod(alphas,axis=0)
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        #calculation for q(y_t|y_{t-1})
        self.register_buffer('gammas',to_torch(gammas))
        self.register_buffer('sqrt_one_minus_gammas',to_torch(np.sqrt(1-gammas)))
        self.register_buffer('sqrt_gammas',to_torch(np.sqrt(gammas)))
    
    def noisy_image(self,t,y):
        ''' Compute y_noisy according to (6) p15 of [2]'''
        noise = torch.randn_like(y)
        y_noisy = extract(self.gammas,t,y.shape)*y + extract(self.sqrt_one_minus_gammas,t,noise.shape)*noise
        return y_noisy, noise
        
    def noise_prediction(self,denoise_fn,y_noisy,x,t):
        ''' Use the NN to predict the noise added between y_{t-1} and y_t'''
        noise_pred = denoise_fn(y_noisy,x,t)
        return(noise_pred)