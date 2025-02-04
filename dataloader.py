import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import re


class_dict = {'class_0': 0, 'class_1': 1, 'class_2': 2, 'class_3': 3}
class_names = list(class_dict.keys())


class LaminasAlbDS2(Dataset):
    
    def __init__(self, df, root_dir, transform = None, img_format = True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.r = torchvision.transforms.PILToTensor()
        self.n = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.img_format = img_format
        self.read_fn = Image.open
        if not(self.img_format):
            self.read_fn = torch.load
            self.r = lambda x:x
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        filename1 = (sample["Files"]).split("'")[1]
        filename2 = (sample["Files"]).split("'")[3]
        meta = self.df.iloc[idx,3:].astype(np.float32).values
        if not(self.img_format):
            filename1 = filename1.split('.')[0] + '.pt'
            filename2 = filename2.split('.')[0] + '.pt'

        label = class_dict[sample["target"]]
        try:
            img1 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename1)))/1.)
            img2 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename2)))/1.)
            meta = torch.tensor(meta)
            
        except Exception as e:
            print(e)
            return None
        
        if self.transform:
            img1, img2 = img1.numpy(), img2.numpy()
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            img = self.transform(image=img1, image0=img2)
            img1 = torch.from_numpy(np.transpose(img['image'], (2, 0, 1)))
            img2 = torch.from_numpy(np.transpose(img['image0'], (2, 0, 1)))
            
        img = torch.cat((img1, img2), 0)
        return [img, label, sample['Nome'], meta]

    
class LaminasTextDS(Dataset):
    
    def __init__(self, df, root_dir, norm_fn, transform = None, text_transform = None, img_format = True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.text_transform = text_transform
        self.r = torchvision.transforms.PILToTensor()
        self.n = norm_fn
        
        self.img_format = img_format
        self.read_fn = Image.open
        if not(self.img_format):
            self.read_fn = torch.load
            self.r = lambda x:x
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        filename1 = (sample["Files"]).split("'")[1]
        filename2 = (sample["Files"]).split("'")[3]
        meta = sample['sentence']
        if not(self.img_format):
            filename1 = filename1.split('.')[0] + '.pt'
            filename2 = filename2.split('.')[0] + '.pt'

        label = class_dict[sample["target"]]
        try:
            img1 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename1)))/1.)
            img2 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename2)))/1.)
            
        except Exception as e:
            return None
        
        if self.text_transform:
            meta = self.text_transform(meta)
        
        if self.transform:
            img1, img2 = img1.numpy(), img2.numpy()
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            img = self.transform(image=img1, image0=img2)
            img1 = torch.from_numpy(np.transpose(img['image'], (2, 0, 1)))
            img2 = torch.from_numpy(np.transpose(img['image0'], (2, 0, 1)))
            
        img = torch.cat((img1, img2), 0)
        return [img, label, sample['Nome'], meta]


class LaminasTextDS_v2(Dataset):
    """
    Dataset for full sentences, text transformation made at the start of the epoch
    """
    def __init__(self, df, root_dir, norm_fn, transform = None, text_transform = None, img_format = True, max_length=12):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.text_transform = text_transform
        self.r = torchvision.transforms.PILToTensor()
        self.n = norm_fn
        self.max_length = max_length

        self.img_format = img_format
        self.read_fn = Image.open
        if not(self.img_format):
            self.read_fn = torch.load
            self.r = lambda x:x
        
    def __len__(self):
        return len(self.df)
    
    
    @staticmethod
    def truncate(s:str, limit:int)->str:
        count = 0
        new_s = " ".join(s.split()[:(limit+int(0.3*limit))])
        pattern = r"[,.(]"
        match = re.search(pattern, new_s[::-1])
        if match:
            return new_s[:-(match.start()+1)]
        else:
            return new_s
    
    @staticmethod
    def basic_clean(s:str)->str:
        pattern = r'(?<=\S)/(?=\S)'
        replacement = " ou "
        return re.sub(pattern, replacement, s)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        filename1 = (sample["Files"]).split("'")[1]
        filename2 = (sample["Files"]).split("'")[3]
        meta = sample['sentence']
        meta = LaminasTextDS_v2.truncate(meta, self.max_length)
        meta = LaminasTextDS_v2.basic_clean(meta)

        if self.text_transform:
            meta = self.text_transform(meta)

        if not(self.img_format):
            filename1 = filename1.split('.')[0] + '.pt'
            filename2 = filename2.split('.')[0] + '.pt'

        label = class_dict[sample["target"]]
        try:
            img1 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename1)))/1.)
            img2 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename2)))/1.)
            
        except Exception as e:
            return None

        if self.transform:
            img1, img2 = img1.numpy(), img2.numpy()
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            img = self.transform(image=img1, image0=img2)
            img1 = torch.from_numpy(np.transpose(img['image'], (2, 0, 1)))
            img2 = torch.from_numpy(np.transpose(img['image0'], (2, 0, 1)))
            
        img = torch.cat((img1, img2), 0)
        return [img, label, sample['Nome'], meta]


class LaminasTextDSVAR(Dataset):
    
    def __init__(self, df, root_dir, norm_fn, transform = None, img_format = True, size = 448):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.r = torchvision.transforms.PILToTensor()
        self.n = norm_fn
        
        self.img_format = img_format
        self.read_fn = Image.open
        if not(self.img_format):
            self.read_fn = torch.load
            self.r = torchvision.transforms.CenterCrop(self.size)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        filename1 = (sample["Files"]).split("'")[1]
        filename2 = (sample["Files"]).split("'")[3]
        meta = sample['sentence']
        if not(self.img_format):
            filename1 = filename1.split('.')[0] + '.pt'
            filename2 = filename2.split('.')[0] + '.pt'

        label = class_dict[sample["target"]]
        try:
            img1 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename1)))/1.)
            img2 = self.n(self.r(self.read_fn(os.path.join(self.root_dir, filename2)))/1.)
            
        except Exception as e:
            return None
        
        if self.transform:
            img1, img2 = img1.numpy(), img2.numpy()
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            img = self.transform(image=img1, image0=img2)
            img1 = torch.from_numpy(np.transpose(img['image'], (2, 0, 1)))
            img2 = torch.from_numpy(np.transpose(img['image0'], (2, 0, 1)))
            
        img = torch.cat((img1, img2), 0)
        return [img, label, sample['Nome'], meta]