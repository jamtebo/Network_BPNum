import torch
import torchvision.transforms as transforms 
import numpy as np
import os
from PIL import Image
from ImageOperate import*

class imageDataSet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(imageDataSet, self).__init__()
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root,k) for k in imgs]
        self.transform = transform

    def __getitem__(self,index):
        img_path = self.imgs[index]
        pil_img =Image.open(img_path)
        img_data = GetImageData(28, 28, img_path)
        img_data = img_data.reshape(1, 28, 28)
        data = torch.from_numpy(img_data)
        return data

    def __len__(self):
        return len(self.imgs)  
