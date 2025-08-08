import numpy as np
from imageio.v2 import imread
import torch
from torch.utils.data import Dataset
import os
from skimage import transform
from scipy import ndimage
import random

class DistortDL(Dataset):
    def __init__(self, 
        data_path, img_size
    ):
        self.data_path = data_path
        self.img_list = os.listdir(data_path)
        for i, f in enumerate(self.img_list): self.img_list[i] = os.path.join(data_path, f)
        self.img_size = img_size
        return
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img = np.float32(imread(self.img_list[idx]))[..., :3] / 255.0
        
        if img.shape[0] < self.img_size or img.shape[1] < self.img_size:
            img = transform.resize(
                image=np.float32(img[:, :, :3]), 
                output_shape=(self.img_size, self.img_size),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True,
            )
        
        rand_h = random.randint(0, img.shape[0] - self.img_size)
        rand_w = random.randint(0, img.shape[1] - self.img_size)
        
        img = img[rand_h:rand_h + self.img_size, rand_w:rand_w + self.img_size, :] * 255.
        
        rand_distortion = random.randint(0, 2)
        
        mode = torch.ones((1, self.img_size, self.img_size))

        if rand_distortion == 0:
            img_ = self.AWGN(img)
            mode = mode * 0.0
        elif rand_distortion == 1:
            img_ = self.GaussianBlur(img)
        elif rand_distortion == 2:
            img_ = self.Decolorize(img)
            mode = mode * 2.0
        
        return torch.from_numpy(img_).permute(2, 0, 1), torch.from_numpy(img).permute(2, 0, 1), mode    

    # Add white noise to the image
    def AWGN(self, img, noise_range = (10, 50)):
        noise_level = random.uniform(noise_range[0], noise_range[1])
        noisy = np.float32(np.clip(img + np.random.normal(0, noise_level, img.shape), 0., 255.))
        return noisy
    
    # Apply Gaussian blur to the image
    def GaussianBlur(self, patch, blur_range = (0.5, 2)):
        sigma = random.uniform(blur_range[0], blur_range[1])
        patch[..., 0] = np.clip(ndimage.gaussian_filter(patch[..., 0], sigma = sigma), 0., 255.)
        patch[..., 1] = np.clip(ndimage.gaussian_filter(patch[..., 1], sigma = sigma), 0., 255.)
        patch[..., 2] = np.clip(ndimage.gaussian_filter(patch[..., 2], sigma = sigma), 0., 255.)
        return patch
    
    # Convert the image to grayscale
    def Decolorize(self, patch):
        patch_ = np.zeros_like(patch)
        y = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
        patch_[..., 0], patch_[..., 1], patch_[..., 2] = y, y, y 
        return patch_


class DenoiseDL(Dataset):
    def __init__(self, 
        data_path, img_size
    ):
        self.data_path = data_path
        self.img_list = os.listdir(data_path)
        for i, f in enumerate(self.img_list): self.img_list[i] = os.path.join(data_path, f)
        self.img_size = img_size
        return
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img = np.float32(imread(self.img_list[idx]))[..., :3] / 255.0
        
        if img.shape[0] < self.img_size or img.shape[1] < self.img_size:
            img = transform.resize(
                image=np.float32(img[:, :, :3]), 
                output_shape=(self.img_size, self.img_size),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True,
            )
        
        rand_h = random.randint(0, img.shape[0] - self.img_size)
        rand_w = random.randint(0, img.shape[1] - self.img_size)
        
        img = img[rand_h:rand_h + self.img_size, rand_w:rand_w + self.img_size, :] * 255.
        img_ = self.AWGN(img)
        
        return torch.from_numpy(img_).permute(2, 0, 1), torch.from_numpy(img).permute(2, 0, 1)   

    def AWGN(self, img, noise_range = (10, 60)):
        noise_level = random.uniform(noise_range[0], noise_range[1])
        noisy = np.float32(np.clip(img + np.random.normal(0, noise_level, img.shape), 0., 255.))
        return noisy

