# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 03:42:02 2018

@author: jinq
"""

import os

import numpy as np

import torch

import pretrainedmodels.utils as putils

def GetLatestFile(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def FindSubstring(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ''

class ReadCSV(object):    
    def __init__(self, csv_path):
        img_name = np.genfromtxt(csv_path, delimiter=',', usecols=0, 
                                 dtype=str, skip_header=1).astype(object)
        onehot_labels = np.genfromtxt(csv_path, delimiter=',',
                                      skip_header=1)[:, 1:]
        labels = np.argmax(onehot_labels, axis=1).astype(object)
        self.label_list = np.genfromtxt(csv_path, delimiter=',',
                                        dtype=str)[0, 1:]
        self.data = np.vstack([img_name, labels])

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, root, datalist, transform=None, target_transform=None,
                 space='RGB'):
        self.root = root
        self.datalist = datalist
        self.transform = transform
        self.target_transform = target_transform
        self.loader = putils.LoadImage(space)
    
    def __getitem__(self, index):
        img_file, label = self.datalist[index]
        img = self.loader('{}.jpg'.format(os.path.join(self.root, img_file)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
    
    def __len__(self):
        return len(self.datalist)

def main():
    path = '../data'
    print(GetLatestFile(path))
    a = '/scratch/group/20180602200639_Acc_0.2346.pt'
    b = float(FindSubstring(a, '_Acc_', '.pt'))
    print(b, 1-b, b < 1)

if __name__ == '__main__':
    main()
