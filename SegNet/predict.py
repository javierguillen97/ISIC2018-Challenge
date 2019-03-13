import argparse
import LoadBatches
import FCN32
import FCN8
import VGGSegnet
import VGGUnet

from keras.optimizers import RMSprop
from keras.models import load_model

import glob
import cv2
import numpy as np
import random


def jaccard_distance(y_true, y_pred, smooth=0):
	"""Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
	This loss is useful when you have unbalanced numbers of pixels within an image
	because it gives all classes equal weight. However, it is not the defacto
	standard for image segmentation.
	For example, assume you are trying to predict if each pixel is cat, dog, or background.
	You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
	should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
	The loss has been modified to have a smooth gradient as it converges on zero.
	This has been shifted so it converges on 0 and is smoothed to avoid exploding
	or disappearing gradient.
	Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
					= sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
	# References
	Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
	What is a good evaluation measure for semantic segmentation?.
	IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
	https://en.wikipedia.org/wiki/Jaccard_index
	"""
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
	jac = (intersection + smooth) / (sum_ - intersection + smooth)
	return (1 - jac) * smooth

parser = argparse.ArgumentParser()
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

weights_path = "weights/isic_task1/final_segnet.h5"

# n_classes = 10
n_classes = 2

model_name = "vgg_segnet"

# images_path = "dataset1/images_prepped_test/"
images_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1-2_Input/"

input_width =  224
input_height = 224

modelFns = {'vgg_segnet':VGGSegnet.VGGSegnet, 'vgg_unet':VGGUnet.VGGUnet, 'vgg_unet2':VGGUnet.VGGUnet2, 'fcn8':FCN8.FCN8, 'fcn32':FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(weights_path)

rmsprop = RMSprop(lr=0.001, decay=1e-6, epsilon=0.1)
m.compile(loss='categorical_crossentropy',
      optimizer= rmsprop,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") +  glob.glob(images_path + "*.jpeg")
images.sort()

# colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(n_classes)]
colors = [(0, 0, 0), (255, 255, 255)]

output_path = "result/"
total_jaccard = 0
for imgName in images:
	outName = imgName.replace(images_path, output_path)
	X = LoadBatches.getImageArr(imgName, input_width, input_height)
	pr = m.predict( np.array([X]) )[0]

	ground_truth = getSegmentationArr(imgName, 2, input_width, input_height)

	total_jaccard += jaccard_distance(pr, ground_truth)

	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

	seg_img = np.zeros((output_height, output_width, 3))
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	cv2.imwrite(  outName , seg_img )
