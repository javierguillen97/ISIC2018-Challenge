# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from keras import backend as K
from keras.utils import np_utils
from numpy import genfromtxt as gft

import cv2
import csv
import re
import numpy as np

from os import listdir

nb_train_samples=5
#num_classes=7

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

def read_image_list():
	DATA_DIR='/home/grads/k/kaihe/Documents/ISIC/Task3/'
	TASK3_DATA_DIR=DATA_DIR+'ISIC2018_Task3_Training_Input/Image/'

	TASK3_TRUTH_FILE = DATA_DIR + 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

	pattern = re.compile(r"\w+\.jpg")

	image_files = []
	truth_file = open(TASK3_TRUTH_FILE)
	csv_reader = csv.reader(truth_file)
	for row in csv_reader:
		image_files.append(TASK3_DATA_DIR + row[0] + '.jpg')
	image_files = image_files[1:]

	labels = gft(TASK3_TRUTH_FILE, delimiter=',')
	labels = labels[1:, 1:]

	return image_files, labels 

def load_images(image_list, img_rows, img_cols):
	images = np.empty((len(image_list), img_rows, img_cols, 3))
	# Memory error !
	#images=np.empty((nb_train_sample,img_rows,img_cols,3))
	i = 0
	for f in image_list:
		image = cv2.imread(f)
		#image = cv2.resize(image, (img_rows, img_cols)) 
		if K.image_dim_ordering() == 'th':
			image=np.array([cv2.resize(image.transpose(1,2,0),(img_rows,img_cols)).transpose(2,0,1)])
		else: 
			image=np.array([cv2.resize(image,(img_rows,img_cols))])

		images[i,:,:,:] = image
		i+=1
		
	# Switch RGB to BGR order
	images = images[:, :, :, ::-1]

	# Subtract ImageNet mean pixel
	images[:, :, :, 0] -= 103.939
	images[:, :, :, 1] -= 116.779
	images[:, :, :, 2] -= 123.68

	return images

def load_ISIC_data(img_rows,img_cols,validation_size):
	
	
	image_list, Y_all=read_image_list()
	X=load_images(image_list,img_rows,img_cols)
	
	X_train=X[:int(round(X.shape[0]*(1-validation_size))),:,:,:]
	X_valid=X[int(round(X.shape[0]*(1-validation_size))):,:,:,:]
	
	Y_train=Y_all[:int(round(Y_all.shape[0]*(1-validation_size))),:]
	Y_valid=Y_all[int(round(Y_all.shape[0]*(1-validation_size))):,:]
	
	return X_train, Y_train, X_valid, Y_valid

	# convert string into tensors
	#all_images=ops.convert_to_tensor(image_list,dtype=dtypes.string)
	#all_labels=ops.convert_to_tensor(labels,dtype=dtypes.int32)

	# create a partition vector
	#partitions = [0] * len(image_list)
	#partitions[:test_set_size] = [1] * test_set_size
	#random.shuffle(partitions)
	
	# partition our data into a test and train set according to our partition vector
	#train_images, test_images = tf.dynamic_partition(X, partitions, 2)
	#train_labels, test_labels = tf.dynamic_partition(labels, partitions, 2)
	
	# create input queues
	#train_input_queue = tf.train.slice_input_producer(
         #                           	[train_images, train_labels],
          #                          	shuffle=False)
	#test_input_queue = tf.train.slice_input_producer(
         #                           	[test_images, test_labels],
         #                           	shuffle=False)

	# process path and string tensor into an image and a label
	#file_content = tf.read_file(train_input_queue[0])
	#train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
	#train_label = train_input_queue[1]

	#file_content = tf.read_file(test_input_queue[0])
	#test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
	#test_label = test_input_queue[1]
	
	#return X_train, Y_train, X_valid, Y_valid

	# define tensor shape
	#train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
	#test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


	# collect batches of images before processing
	#train_image_batch, train_label_batch = tf.train.batch(
                                    	#[train_image, train_label],
                                    	#batch_size=BATCH_SIZE
                                    	#,num_threads=1
                                    #)
	#test_image_batch, test_label_batch = tf.train.batch(
                                   	#[test_image, test_label],
                                    	#batch_size=BATCH_SIZE
                                    	#,num_threads=1
                                    	#)

#print "input pipeline ready"

