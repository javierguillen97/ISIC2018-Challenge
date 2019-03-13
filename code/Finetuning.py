# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:29:12 2018

@author: jinq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import os
import time
from datetime import datetime, timedelta
import copy
import numpy as np

from utils import GetLatestFile, FindSubstring, ReadCSV, ImageFilelist

from sklearn.model_selection import StratifiedShuffleSplit

import pretrainedmodels
import pretrainedmodels.utils as putils

from tqdm import tqdm
import torchnet as tnt
from torchnet.engine import Engine

from tensorboardX import SummaryWriter

def train(model, device, train_loader, criterion, optimizer, scheduler):
    scheduler.step()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print('Training Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss +=  loss.item()
            val_corrects += torch.sum(preds == labels.data)
    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
    return val_acc

def main():
    csv_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task3_Training_GroundTruth'
    csv_file = 'ISIC2018_Task3_Training_GroundTruth.csv'
    csv = ReadCSV(os.path.join(csv_path, csv_file))
    
    img_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task3_Training_Input'
    
    data_file = csv.data.T
    
    n_classes = len(np.unique(data_file[:, 1]))
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_fileset = {}
    for train_idx, val_idx in split.split(data_file, data_file[:, 1]):
        strat_fileset['train'] = data_file[train_idx]
        strat_fileset['val'] = data_file[val_idx]
    
    batch_sizes = {'train': 10, 'val': 10}
    num_epochs = 10
    
    model_list = ['vgg19_bn']#,
                  #'vgg11', 'vgg13', 'vgg16', 'vgg19']
#    models_loaded = [pretrainedmodels.__dict__[mdl](num_classes=1000,
#                     pretrained='imagenet') for mdl in model_list]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    result_path = '/scratch/user/jinqing/ISIC2018/results'
    
    for mdl_name in tqdm(model_list):
        mdl = pretrainedmodels.__dict__[mdl_name](num_classes=1000,
                                                  pretrained='imagenet')
        since = time.time()
        
        dim_features = mdl.last_linear.in_features
        mdl.last_linear = nn.Linear(dim_features, n_classes)
        mdl = mdl.to(device)
        
        tf_img = putils.TransformImage(mdl)
        
        strat_datasets = {k: ImageFilelist(img_path, strat_fileset[k], tf_img)
                            for k in ['train', 'val']}
        
        strat_dataloader = {k: torch.utils.data.DataLoader(strat_datasets[k],
                                                           batch_size=batch_sizes[k],
                                                           shuffle=True,
                                                           num_workers=4)
                            for k in ['train', 'val']}
        
        best_acc = 0.0
        
        mdl_path = os.path.join(result_path, mdl_name)
        if os.path.isdir(mdl_path):
            mdl_file = GetLatestFile(mdl_path)
            mdl.load_state_dict(torch.load(mdl_file))
            best_acc = FindSubstring(mdl_file, '_Acc_', '.pt')
            best_acc = float(best_acc)
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer_ft = optim.SGD(mdl.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                                     step_size=7,
                                                     gamma=0.1)
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            train(mdl, device, strat_dataloader['train'], criterion,
                  optimizer_ft, exp_lr_scheduler)
            val_acc = validate(mdl, device, strat_dataloader['val'], criterion)
            if val_acc > best_acc:
                best_acc = val_acc
                best_weight_file = '{:%Y%m%d%H%M%S}_Acc_{:.4f}.pt'.format(
                        datetime.now(), best_acc)
                if not os.path.isdir(mdl_path):
                    os.makedirs(mdl_path)
                best_weight_file = os.path.join(mdl_path, best_weight_file)
                torch.save(mdl.state_dict(), best_weight_file)
        
        t_d = timedelta(seconds=time.time() - since)
        t_d = [int(float(x)) for x in str(t_d).split(':')]
        print('Time Elapsed for {}: {}:{}:{}'.format(mdl_name, *t_d))

if __name__ == '__main__':
    main()
