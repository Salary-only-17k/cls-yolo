import os
import pathlib
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DataLoader



class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def transfroms(input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.CenterCrop((input_size, input_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

class Dataset_1head(data.Dataset): 
    def __init__(self, mode, dir:str,format:str='.jpg'):  
        self.mode = mode
        self.format =format
        self.list_img = []  
        self.list_label= [] 
        self.transform = transfroms()  
        assert (self.mode in ['train', 'val']),ValueError
        dir = os.path.join(dir,self.mode) #!^
        for filepth in list(pathlib.Path(dir).glob(f"*{self.format}")):  #!^
            filepth = str(filepth)
            self.list_img.append(filepth)  
            filename = os.path.basename(filepth)
            label_pool = filename.split('_')
            label = label_pool[1]  #!^
            self.list_label.append(float(label))    #!^ 
        self.data_size = len(self.list_label)       
       
    def __doc__(self):
        print("0000n_Alabel_.jpg")

    def __getitem__(self, item):  
        if self.mode == 'train':  
            img = Image.open(self.list_img[item]) 
            label = self.list_label[item]  
            return self.transform['train'](img),torch.LongTensor([label])  
        elif self.mode in ['val', 'test']:  
            img = Image.open(self.list_img[item])  
            label = self.list_label[item]   #!^
            return self.transform[self.mode](img), torch.LongTensor([label])  #!^
        else:
            print('None')

    def __len__(self):
        return self.data_size  # 返回数据集大小

