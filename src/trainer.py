import os

from .network import UNetModel,EMA
from .dataloader import gray_color_data
from diffusion import GaussianDiffusion,extract

import torch
import torch.optim as optim
import torch.nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataloader

import numpy as np

class Trainer(nn.Module):
    def __init__(self,config):
        super().__init__()
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
        dataset_train = gray_color_data(self.path_train_color,self.path_train_grey)  ## Refaire pour renvoyer du rgb
        dataset_validation = gray_color_data(self.path_validation_color,self.path_validation_grey)
        self.batch_size = config.BATCH_SIZE
        self.dataloader_train = Dataloader(dataset_train,batch_size=self.batch_size, shuffle=True)
        self.dataloader_validation = Dataloader(dataset_validation,batch_size=1,shuffle=False)
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

        def save_model(self,name):
            torch.save(self.network.state_dict(),f'models/{name}')

        def train(self):

                writer = SummaryWriter(log_dir='results/') 
                optimizer = optim.Adam(self.network.parameters(),lr=self.LR)
                iteration = 0

                while iteration < self.iteration_max:
                    
                    self.network.train()
                    optimizer.zero_grad()
                    grey,color = next(self.dataloader_train).to(self.device)
                    t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=device).long()
                    noisy_image,noise_ref = self.diffusion.noisy_image(t,color)
                    noise_pred = self.diffusion.noise_prediction(self.network,noisy_image,grey,t)
                    loss = self.loss(noise_ref,denoised_image)
                    loss.backward()
                    optimizer.step()
                    iteration+=1

                    writer.add_scalar('MSE/Train',loss.item() * self.batch_size,iteration)

                    if iteration%self.ema_every == 0 and iteration>self.start_ema:
                        self.EMA.update_model_average(self.EMA,self.network)
                    
                    if iteration%self.save_model_every == 0:
                        self.save_model(f'model_{iteration}.pth')

                    if iteration%self.validation_every == 0:
                        with torch.no_grad():
                            self.network.eval()
                            for (gray,color) in self.dataloader_validation:
                                gray = gray.to(self.device)
                                T = 1000
                                alphas = np.linspace(1e-4,0.09,T)
                                gammas = np.cumprod(alphas,axis=0)
                                y = torch.randn_like(color).to(self.device)
                                for t in range(T):
                                    if t == 0 :
                                        z = torch.randn_like(color).to(self.device)
                                    else:
                                        z = torch.zeros_like(color).to(self.device)
                                    y = extract(to_torch(np.sqrt(1/alphas)),t,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),t,y.shape))*self.network(y,gray,t)) + extract(np.sqrt(1-alphas),t,z.shape)*z
                                loss = #check metrics for colorization

            

        