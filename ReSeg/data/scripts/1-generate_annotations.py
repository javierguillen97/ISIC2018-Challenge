import os, glob
from PIL import Image
import numpy as np
from scipy.io import loadmat
import cv2

DATA_DIR = '/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/'
METHOD_DIR = '/Users/Ardywibowo/Documents/Projects/ISIC-Challenge/ReSeg/data/'
IMG_DIR = [os.path.join(DATA_DIR, 'ISIC2018_Task1-2_Training_Input/'), os.path.join(DATA_DIR, 'ISIC2018_Task1-2_Validation_Input/')]
OUTPUT_DIR = os.path.join(METHOD_DIR, 'processed', 'annotations')

try:
  os.makedirs(OUTPUT_DIR)
except:
  pass

ann_files = glob.glob(os.path.join(DATA_DIR, 'ISIC2018_Task1_Training_GroundTruth/', '*.png')) + glob.glob(os.path.join(DATA_DIR, 'ISIC2018_Task1_Validation_GroundTruth/', '*.png'))

for ann_file in ann_files:
  img_name = os.path.splitext(os.path.basename(ann_file))[0]
  img_name = img_name.split('_segmentation')[0]
  anns = cv2.imread(ann_file)
  anns = anns[:, :, 0]

  anns[np.where(anns == 255)] = 1

  mask_img = Image.fromarray(anns)
  mask_img.save(os.path.join(OUTPUT_DIR, img_name + '.png'))
