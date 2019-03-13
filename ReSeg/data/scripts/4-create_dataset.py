import os, glob
import numpy as np
from PIL import Image
from utils import create_dataset

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')
RAW_DIR = '/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/'
IMG_DIR = [os.path.join(RAW_DIR, 'ISIC2018_Task1-2_Training_Input/'), os.path.join(RAW_DIR, 'ISIC2018_Task1-2_Validation_Input/')]
OUT_DIR = os.path.join(DATA_DIR, 'processed', 'lmdb')

try:
    os.makedirs(OUT_DIR)
except:
    pass

for subset in ['training', 'test']:
    lst_filepath = os.path.join(DATA_DIR, 'metadata', subset + '.lst')
    lst = np.loadtxt(lst_filepath, dtype='str', delimiter=' ')
    img_paths = []; ann_paths = []
    for image_name in lst:
        img_path = glob.glob(os.path.join(IMG_DIR[0], image_name + '.jpg')) + glob.glob(os.path.join(IMG_DIR[1], image_name + '.jpg'))
        ann_path = os.path.join(ANN_DIR, image_name + '.png')
        img_paths.append(img_path[0])
        ann_paths.append(ann_path)

    out_path = os.path.join(OUT_DIR, '{}-lmdb'.format(subset))

    create_dataset(out_path, img_paths, ann_paths)
