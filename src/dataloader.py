import cv2
import numpy as np
import os
import random
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class gray_color_data(Dataset):
    def __init__(self,path_color,path_gray):
        super().__init__()
        self.path_color = path_color
        self.path_gray = path_gray
        self.data_color = np.load(path_color)
        self.data_gray = np.load(path_gray)
    def __len__(self):
        return len(self.data_color)
    def __getitem__(self,idx):
        image_gray =  self.data_gray[idx]
        shape = (image_gray.shape[0],image_gray.shape[1],3)
        image_color = np.zeros((224,224,3))
        image_color[:,:,0] = image_gray
        image_color[:,:,1:] = self.data_color[idx]
        image_color = image_color.astype('uint8')
        image_color = cv2.cvtColor(image_color,cv2.COLOR_LAB2RGB)
        return(ToTensor()(image_gray),ToTensor()(image_color))