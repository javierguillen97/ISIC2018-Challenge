import sys
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import cv2
import numpy as np
import sys

from pspnet.pspnet import pspnet
from pspnet.metrics import runningScore
running_metrics = runningScore(2)

# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorboardX import SummaryWriter


area = 300 * 300
def get_batch(path, ids, cuda):
    images, masks = np.empty((len(ids), 3, 300, 300)), np.empty((len(ids), 300, 300))
    for i in range(len(ids)):
        id = ids[i]
        image = cv2.imread(path + id + ".jpg") # h, w, c
        image = np.swapaxes(image, 0, 2) # c, w, h
        images[i, :, :, :] = image
        mask = cv2.imread(path + id + "_mask.png")[:, :, 2]
        mask = mask.astype(int)
        masks[i, :, :] = mask / 255.
    if cuda:
        images = Variable(torch.Tensor(images)).cuda()
        masks = Variable(torch.Tensor(masks)).cuda()
    else:
        images = Variable(torch.Tensor(images))
        masks = Variable(torch.Tensor(masks))
    return images, masks



task_name = "pspnet_patch_7_8_2018"


path_train = "/ssd1/chenwy/isic/task1/train/"
path_val = "/ssd1/chenwy/isic/task1/val/"

ids_train = [ img[:-4] for img in os.listdir(path_train) if img.endswith(".jpg") ]
ids_val = [ img[:-4] for img in os.listdir(path_val) if img.endswith(".jpg") ]
print(len(ids_train))
print(len(ids_val))

num_epochs = 500
batch_size = 6
lr = 0.001
cuda = True
pretrain = False

model = pspnet(n_classes=2, input_size=(300, 300))

if cuda:
    model = model.cuda()
    model = nn.DataParallel(model)


optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
criterion = model.loss

if pretrain:
    model.load_state_dict(torch.load("/home/chenwy/isic/saved_models/" + task_name + ".pth"))

writer = SummaryWriter(log_dir="runs/" + task_name)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(range(int(len(ids_train) / batch_size))):
        ids = np.random.choice(ids_train, batch_size, replace = False)
        # print(ids)
        images, masks = get_batch(path_train, ids, cuda)
        outputs = model(images)
        loss = criterion(input=outputs, target=masks)
        # print("loss:", loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print("evaluating...")
        '''
        this evaluation section is to reconstruct image from patches, which could be outdated
        please just focus on the running_metrics usage
        '''
        for id in tqdm(ids_val):
            image = cv2.imread(path_val + id + ".jpg") # h, w, c
            image = np.swapaxes(image, 0, 2) # c, w, h
            mask = cv2.imread(path_val + id + "_mask.png")[:, :, 2] # h, w
            mask = np.swapaxes(mask, 0, 1) # w, h
            mask = mask.astype(int)
            mask = mask / 255.
            shape = image.shape
            w_p, h_p = (300 - image.shape[1] % 300) % 300, (300 - image.shape[2] % 300) % 300
            image = np.pad(image, [(0, 0), (0, w_p), (0, h_p)], mode='constant', constant_values=0)
            row = (shape[1] + w_p) // 300
            column = (shape[2] + h_p) // 300
            patches = np.empty((row * column, 3, 300, 300))
            for i in range(row):
                for j in range(column):
                    patches[i * column + j] = image[:, i * 300: (i + 1) * 300, j * 300: (j + 1) * 300]
            patches = Variable(torch.Tensor(patches)).cuda()
            i = 0
            outputs = None
            while i < patches.size()[0]:
                if patches.size()[0] - i <= 3:
                    _, output = model(torch.cat((patches[i: i + batch_size], patches[i: i + batch_size]), 0))
                    output = output[:output.size()[0] // 2]
                else: _, output = model(patches[i: i + batch_size])
                output = output.data.cpu().numpy()
                if outputs is None: outputs = output
                else: outputs = np.concatenate((outputs, output), axis=0)
                i += batch_size
            prediction = np.empty((2, row * 300, column * 300))
            for i in range(row):
                for j in range(column):
                    prediction[:, i * 300: (i + 1) * 300, j * 300: (j + 1) * 300] = outputs[i * column + j]
            prediction = prediction[:, :shape[1], :shape[2]]
            prediction = prediction.argmax(0)
            running_metrics.update([mask], [prediction])
        
        score, class_iou = running_metrics.get_scores()
        iou_val, score_val = score["IoU"], score["IoU_threshold"]
        running_metrics.reset()
        
        # print('epoch [{}/{}], training loss:{:.4f}, IoU:{:.4f}, score:{:.4f}'.format(epoch+1, num_epochs, loss.data[0], iou_train, score_train))
        print('epoch [{}/{}], validation IoU:{:.4f}, score:{:.4f}'.format(epoch+1, num_epochs, iou_val, score_val)) 
        # writer.add_scalars('Loss', {'training loss': loss.data[0], 'validation loss': loss_val.data[0]}, epoch)
        writer.add_scalars('IoU', {'validation iou': iou_val}, epoch)
        writer.add_scalars('score', {'validation score': score_val}, epoch)
        
        # =============================================
        if cuda: torch.cuda.empty_cache()
        torch.save(model.state_dict(), "./saved_models/" + task_name + ".pth")
