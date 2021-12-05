import torch
import os
import cv2
import yaml
import logging
from .augmentations import RandAugment
import numpy as np 
from .imbalance_data_handle import balance_data
import pandas as pd
LOGGER = logging.getLogger('__main__.'+__name__)

def preprocess(img,img_size,padding=True):
    if padding:
        height,width,_ = img.shape 
        delta = height - width 
        
        if delta > 0:
            img = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =255)
        else:
            img = np.pad(img,[[-delta//2,-delta//2],[0,0],[0,0]], mode='constant',constant_values =255)
    if isinstance(img_size,int):
        img_size = (img_size,img_size)
    return cv2.resize(img,img_size)

class LoadImagesAndLabels(torch.utils.data.Dataset):
    
    def __init__(self, csv, data_folder, img_size, padding, preprocess=False, augment=False,augment_params=None):
        self.csv_origin = csv 
        self.data_folder = data_folder 
        self.augment = augment 
        self.preprocess = preprocess
        self.padding = padding
        self.img_size = img_size
        if augment:
            self.augmenter = RandAugment(augment_params=augment_params)
        # if augment:
            # self.on_epoch_end(n=500)        
        else:
            self.csv =self.csv_origin
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index,):
        item = self.csv.iloc[index]
        path = os.path.join(self.data_folder, item.path)
        assert os.path.isfile(path),f'this image : {path} is corrupted'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            LOGGER.info(f'this image : {path} is corrupted')
        label = item.label
        
        if self.preprocess:
            img = self.preprocess(img, img_size=self.img_size, padding=self.padding)
        if self.augment:
            img = self.augmenter(img)
        img = np.transpose(img, [2,0,1])
        img = img.astype('float32')/255.
        # img = img/255.
        return img,label,path
            
    def on_epoch_end(self,n=500):
        # self.csv = balance_data(csv=self.csv_origin,image_per_epoch=200)
        csv = self.csv_origin
        labels = set(csv.label)
        dfs = []
        for label in labels:
            df = csv[csv.label==label].sample(n=n,replace=True)
            dfs.append(df)
        df = pd.concat(dfs,axis=0)
        df = df.sample(frac=1).reset_index(drop=True)
        self.csv =  df

