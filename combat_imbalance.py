import torch
import os
from utils import GetLatestFile, FindSubstring, ReadCSV, ImageFilelist
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
import cv2
import csv
import random


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image


def flip_image(image):
    image_h = cv2.flip(image,1)
    image_v = cv2.flip(image,0)
    image_hv = cv2.flip(image,-1)

    return image_h, image_v, image_hv


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w//2, h//2)

    M = cv2.getRotationMatrix2D((cX, cY),-angle, 1.0)
    #cos = np.abs(M[0,0])
    #sin = np.abs(M[0,1])

    #nW = int((h*sin)+(w*cos))
    #nH = int((h*cos)+(w*sin))

    #M[0,2] += (nW/2)-cX
    #M[1,2] += (nH/2)-cY

    return  cv2.warpAffine(image, M, (nW, nH))

def writeImg(img, img_n, img_file, augment_gd_file, categ_idx):
    cv2.imwrite(img_file,img)
    rw_wrt = [0.0]*8
    rw_wrt[0] = img_n.replace('.jpg','')
    rw_wrt[categ_idx+1] = 1.0
    #print(rw_wrt)
    with open(augment_gd_file, 'a') as gd_wrt:
       aug_wrt = csv.writer(gd_wrt, delimiter=',')
       aug_wrt.writerow(rw_wrt)

def balance_class(data_list, input_path, out_path, augment_gd_file, categ_idx, flip, brightness, target_size):
    original_size = len(data_list)
    print('The original size is ',original_size)

    size_gap = max(0,target_size-original_size)
    print('The gap is ',size_gap)
    if size_gap ==0:
      aug = 0
      print('Subsampling...')
      np.random.shuffle(data_list)
      for img_n in data_list[:target_size]:
        sub_img = cv2.imread(input_path+img_n)
        writeImg(sub_img, img_n, out_path+img_n, augment_gd_file, categ_idx)
        aug = aug+1
    else:
      print('Augmentation...')
      save_list = []
      aug = 0
      for img_n in data_list:
        sub_img = cv2.imread(input_path+img_n)
        writeImg(sub_img, img_n, out_path+img_n, augment_gd_file, categ_idx)
        save_list.append(img_n)
        aug = aug+1 
      while aug < target_size:
        img_sel = random.choice(data_list)
        img = cv2.imread(input_path+img_sel)
        
        img_b = augment_brightness_camera_images(img)
        img_n = img_sel.replace('.jpg','_b'+'.jpg')
        while img_n in save_list:
          img_n = img_sel.replace('.jpg','_b'+str(np.random.randint(50))+'.jpg')
       
        writeImg(img_b, img_n, out_path+img_n, augment_gd_file, categ_idx)
        save_list.append(img_n)
        aug = aug+1
     
        img_f = flip_image(img)
      
        for i in range(3):
          img_n = img_sel.replace('.jpg','_f'+str(i)+'.jpg')
          if img_n not in save_list:
            writeImg(img_f[i],img_n, out_path+img_n, augment_gd_file, categ_idx)
            save_list.append(img_n)
            aug = aug+1
          if np.random.randint(2)==1:
            img_b = augment_brightness_camera_images(img_f[i])
            img_n = img_sel.replace('.jpg','_bf'+str(i)+'.jpg')
            while img_n in save_list:
              img_n = img_sel.replace('.jpg','_bf'+str(i)+str(np.random.randint(50))+'.jpg')
            writeImg(img_b,img_n, out_path+img_n, augment_gd_file, categ_idx)
            save_list.append(img_n)
            aug = aug+1
     
    print('After augmentation: ',aug)
     # for rt in angle:
        #print(rt)
     #   img_r = rotate_bound(img, rt)
     #   img_n = img_sel.replace('.jpg','_r'+str(rt)+'.jpg')
     #   writeImg(img_r, img_n, out_path+img_n, augment_gd_file, categ_idx)
     #   img_b = augment_brightness_camera_images(img_r)
     #   img_n = img_sel.replace('.jpg','_b'+str(rt)+'.jpg')
     #   writeImg(img_b, img_n, out_path+img_n, augment_gd_file, categ_idx)
     #   aug = aug+2



def main():
  csv_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task3_Training_GroundTruth'
  csv_file = 'ISIC2018_Task3_Training_GroundTruth.csv'
  csv = ReadCSV(os.path.join(csv_path, csv_file))
  img_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task3_Training_Input/'
    
  out_path_m0 = '/scratch/user/kaihe/ISIC/augmented_image/seq_m0/'
  augment_gd_file_m0 = '/scratch/user/kaihe/ISIC/augmented_image/Seq_M0_GroundTruth.csv'

  out_path_m1 = '/scratch/user/kaihe/ISIC/augmented_image/seq_m1/'
  augment_gd_file_m1 = '/scratch/user/kaihe/ISIC/augmented_image/Seq_M1_GroundTruth.csv'

  out_path_m2 = '/scratch/user/kaihe/ISIC/augmented_image/seq_m2/'
  augment_gd_file_m2 = '/scratch/user/kaihe/ISIC/augmented_image/Seq_M2_GroundTruth.csv'

  label = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
 
  data_file = csv.data.T
  label_list = csv.label_list.tolist()
  
  label_concern = ['MEL', 'NV']
  label_indice_concern = [label_list.index(k) for k in label_concern]
  label_indice_other = [label_list.index(k) for k in label_list if not k in label_concern]

  n_classes = len(label_list) 
  data_file_concern = np.array([[d[0], label_indice_concern.index(d[1])] for d in data_file if d[1] in label_indice_concern], dtype=object)
  n_classes_concern = len(np.unique(data_file_concern[:, 1]))
  
  data_file_other = np.array([[d[0], label_indice_other.index(d[1])] for d in data_file if d[1] in label_indice_other], dtype=object)
  n_classes_other = len(np.unique(data_file_other[:, 1]))

  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  strat_fileset = {}
  
  for train_idx, val_idx in split.split(data_file, data_file[:,1]):
      strat_fileset['train'] = data_file[train_idx]
      strat_fileset['val'] = data_file[val_idx]

  #for train_idx, val_idx in split.split(data_file_other, data_file_other[:,1]):
  #    strat_fileset['train'] = data_file_other[train_idx]
  #    strat_fileset['val'] = data_file_other[val_idx]
  #for train_idx, val_idx in split.split(data_file_concern, data_file_concern[:,1]):
  #    strat_fileset['train'] = data_file_concern[train_idx]
  #    strat_fileset['val'] = data_file_concern[val_idx]

  for rw in strat_fileset['val']:
      img_n = rw[0]
      img = cv2.imread(img_path+img_n+'.jpg')
      writeImg(img,img_n,out_path_m0+img_n+'.jpg', augment_gd_file_m0, rw[1])
   
  MEL = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='MEL']
  NV = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='NV']
  BCC = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='BCC']
  AKIEC = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='AKIEC']
  BKL = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='BKL']
  DF = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='DF']
  VASC = [rw[0]+'.jpg' for rw in strat_fileset['train'] if label[rw[1]]=='VASC']

  balance_class(MEL, img_path, out_path_m0, augment_gd_file_m0, label.index('MEL'), 1, 1, 1000)
  balance_class(NV, img_path, out_path_m0, augment_gd_file_m0, label.index('NV'), 1, 1, 1000)
  
  balance_class(BCC, img_path, out_path_m0, augment_gd_file_m0, label.index('BCC'), 1, 1, 1000)
  balance_class(AKIEC, img_path, out_path_m0, augment_gd_file_m0, label.index('AKIEC'), 1, 1, 1000)
  balance_class(BKL, img_path, out_path_m0, augment_gd_file_m0, label.index('BKL'), 1, 1, 1000)
  balance_class(DF, img_path, out_path_m0, augment_gd_file_m0, label.index('DF'), 1, 1, 1000)
  balance_class(VASC, img_path, out_path_m0, augment_gd_file_m0, label.index('VASC'), 1, 1, 1000)

if __name__ == '__main__':
    main()


  
  
