import os, glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')
RAW_DIR = '/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/'
OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata')

annotation_files = glob.glob(os.path.join(ANN_DIR, '*.png'))

image_shapes = []
for f in annotation_files:
    image_name = os.path.splitext(os.path.basename(f))[0]
    ann_size = Image.open(f).size
    image_shapes.append([image_name, ann_size[0], ann_size[1]])

np.savetxt(os.path.join(OUTPUT_DIR, 'image_shapes.txt'), image_shapes, fmt='%s', delimiter=',')
