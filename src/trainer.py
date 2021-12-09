import os

from .network import UNetModel,EMA
from .dataloader import gray_color_data
from .diffusion import GaussianDiffusion,extract

import torch
import torch.optim as optim
import torch.nn as nn

#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import tqdm
import copy

class Trainer():
    def __init__(self,config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.diffusion = GaussianDiffusion(config.IMAGE_SIZE,config.CHANNEL_X,config.CHANNEL_Y,config.TIMESTEPS)
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.network = UNetModel(
            config.IMAGE_SIZE,
            in_channels,
            config.MODEL_CHANNELS,
            out_channels,
            config.NUM_RESBLOCKS,
            config.ATTENTION_RESOLUTIONS,
            config.DROPOUT,
            config.CHANNEL_MULT,
            config.CONV_RESAMPLE,
            config.USE_CHECKPOINT,
            config.USE_FP16,
            config.NUM_HEADS,
            config.NUM_HEAD_CHANNELS,
            config.NUM_HEAD_UPSAMPLE,
            config.USE_SCALE_SHIFT_NORM,
            config.RESBLOCK_UPDOWN,
            config.USE_NEW_ATTENTION_ORDER,
            ).to(self.device)
        self.path_train_color = os.path.join(config.PATH_COLOR,'train.npy')
        self.path_train_grey = os.path.join(config.PATH_GREY,'train.npy')
        self.path_validation_color = os.path.join(config.PATH_COLOR,'validation.npy')
        self.path_validation_grey = os.path.join(config.PATH_GREY,'validation.npy')
        dataset_train = gray_color_data(self.path_train_color,self.path_train_grey)  
        dataset_validation = gray_color_data(self.path_validation_color,self.path_validation_grey)
        self.batch_size = config.BATCH_SIZE
        self.dataloader_train = DataLoader(dataset_train,batch_size=self.batch_size, shuffle=True)
        self.dataloader_validation = DataLoader(dataset_validation,batch_size=1,shuffle=False)
        self.iteration_max = config.ITERATION_MAX
        self.EMA = EMA(0.9999)
        self.LR = config.LR
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        if config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else :
            print('Loss not implemented, setting the loss to L2 (default one)')
        self.num_timesteps = config.TIMESTEPS
        self.validation_every = config.VALIDATION_EVERY
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.ema_model = copy.deepcopy(self.network).to(self.device)

    def save_model(self,name,EMA=False):
        if not EMA:
            torch.save(self.network.state_dict(),f'models/{name}')
        else:
            torch.save(self.ema_model.state_dict(),f'models/{name}')

    def train(self):

            optimizer = optim.Adam(self.network.parameters(),lr=self.LR)
            iteration = 0
            
            print('Starting Training')

            while iteration < self.iteration_max:

                tq = tqdm(self.dataloader_train)
                
                for grey,color in tq:
                    tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                    self.network.train()
                    optimizer.zero_grad()

                    t = torch.randint(0, self.num_timesteps, (self.batch_size,)).long()
                    noisy_image,noise_ref = self.diffusion.noisy_image(t,color)
                    noise_pred = self.diffusion.noise_prediction(self.network,noisy_image.to(self.device),grey.to(self.device),t.to(self.device))
                    loss = self.loss(noise_ref,denoised_image)
                    loss.backward()
                    optimizer.step()
                    tq.set_postfix(loss = loss.item()*self.batch_size)
                    
                    iteration+=1

                    if iteration%self.ema_every == 0 and iteration>self.start_ema:
                        print('EMA update')
                        self.EMA.update_model_average(self.ema_model,self.network)

                    if iteration%self.save_model_every == 0:
                        print('Saving models')
                        self.save_model(f'model_{iteration}.pth')
                        self.save_model(f'model_ema_{iteration}.pth',EMA=True)

                    if iteration%self.validation_every == 0:
                        tq_val = tqdm(self.dataloader_validation)
                        with torch.no_grad():
                            self.network.eval()
                            for grey,color in tq_val:
                                tq.set_description(f'Iteration {iteration} / {iteration_max}')
                                T = 1000
                                alphas = np.linspace(1e-4,0.09,T)
                                gammas = np.cumprod(alphas,axis=0)
                                y = torch.randn_like(color).to(self.device)
                                for t in range(T):
                                    if t == 0 :
                                        z = torch.randn_like(color).to(self.device)
                                    else:
                                        z = torch.zeros_like(color).to(self.device)
                                    y = extract(to_torch(np.sqrt(1/alphas)),t,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),t,y.shape))*self.network(y.to(self.device),grey.to(self.device),t.to(self.device))) + extract(np.sqrt(1-alphas),t,z.shape)*z
                                    y_ema = extract(to_torch(np.sqrt(1/alphas)),t,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),t,y.shape))*self.ema_model(y.to(self.device),grey.to(self.device),t.to(self.device))) + extract(np.sqrt(1-alphas),t,z.shape)*z
                                loss = self.loss(color,y)
                                loss_ema = self.loss(color,y_ema)
                                tq.set_postfix(loss = loss.item())
                    

            

        