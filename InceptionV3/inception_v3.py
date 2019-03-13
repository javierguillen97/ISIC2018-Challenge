# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.models import Model
from keras.utils.vis_utils import plot_model

from keras.optimizers import SGD
from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data
from numpy import genfromtxt

import cv2
import csv
import re
import numpy as np

from os import listdir
from os.path import isfile, join
from os.path import expanduser

def conv2d_bn(x, nb_filter, nb_row, nb_col,
			  border_mode='same', subsample=(1, 1),
			  name=None):
	"""
	Utility function to apply conv + BN for Inception V3.
	"""
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	bn_axis = 1
	x = Convolution2D(nb_filter, nb_row, nb_col,
					  subsample=subsample,
					  activation='relu',
					  border_mode=border_mode,
					  name=conv_name)(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
	return x

def inception_v3_model(img_rows, img_cols, channel=1, num_classes=None):
	"""
	Inception-V3 Model for Keras
	Model Schema is based on 
	https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
	ImageNet Pretrained Weights 
	https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels.h5
	Parameters:
	  img_rows, img_cols - resolution of inputs
	  channel - 1 for grayscale, 3 for color 
	  num_classes - number of class labels for our classification task
	"""
	channel_axis = 1
	img_input = Input(shape=(channel, img_rows, img_cols))
	x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
	x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
	x = conv2d_bn(x, 64, 3, 3)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
	x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# mixed 0, 1, 2: 35 x 35 x 256
	for i in range(3):
		branch1x1 = conv2d_bn(x, 64, 1, 1)

		branch5x5 = conv2d_bn(x, 48, 1, 1)
		branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

		branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
		x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(i))

	# mixed 3: 17 x 17 x 768
	branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
							 subsample=(2, 2), border_mode='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = merge([branch3x3, branch3x3dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed3')

	# mixed 4: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 128, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 128, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed4')

	# mixed 5, 6: 17 x 17 x 768
	for i in range(2):
		branch1x1 = conv2d_bn(x, 192, 1, 1)

		branch7x7 = conv2d_bn(x, 160, 1, 1)
		branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
		branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		branch7x7dbl = conv2d_bn(x, 160, 1, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 192, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 160, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed7')

	# mixed 8: 8 x 8 x 1280
	branch3x3 = conv2d_bn(x, 192, 1, 1)
	branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
						  subsample=(2, 2), border_mode='valid')

	branch7x7x3 = conv2d_bn(x, 192, 1, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
							subsample=(2, 2), border_mode='valid')

	branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
	x = merge([branch3x3, branch7x7x3, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed8')

	# mixed 9: 8 x 8 x 2048
	for i in range(2):
		branch1x1 = conv2d_bn(x, 320, 1, 1)

		branch3x3 = conv2d_bn(x, 384, 1, 1)
		branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
		branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
		branch3x3 = merge([branch3x3_1, branch3x3_2],
						  mode='concat', concat_axis=channel_axis,
						  name='mixed9_' + str(i))

		branch3x3dbl = conv2d_bn(x, 448, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
		branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
		branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
		branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
							 mode='concat', concat_axis=channel_axis)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(9 + i))

	# Fully Connected Softmax Layer
	x_fc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
	x_fc = Flatten(name='flatten')(x_fc)
	x_fc = Dense(1000, activation='softmax', name='predictions')(x_fc)

	# Create model
	model = Model(img_input, x_fc)

	# Load ImageNet pre-trained data 
	model.load_weights('model/inception_v3_weights_th_dim_ordering_th_kernels.h5')

	# Truncate and replace softmax layer for transfer learning
	# Cannot use model.layers.pop() since model is not of Sequential() type
	# The method below works since pre-trained weights are stored in layers but not in the model
	x_newfc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
	x_newfc = Flatten(name='flatten')(x_newfc)
	x_newfc = Dense(num_classes, activation='softmax', name='predictions')(x_newfc)

	# Create another model with our customized softmax
	model = Model(img_input, x_newfc)

	# Learning rate is changed to 0.001
	# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	rmsprop = RMSprop(lr=0.001, decay=1e-6, epsilon=0.1)
	model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

	return model 

def read_images_from_disk(input_queue):
	"""Consumes a single filename and label as a ' '-delimited string.
	Args:
	filename_and_label_tensor: A scalar string tensor.
	Returns:
	Two tensors: the decoded image, and the string label.
	"""

	label = input_queue[1]
	file_contents = tf.read_file(input_queue[0])
	example = tf.image.decode_png(file_contents, channels=3)
	return example, label

def generate_data(file_list, labels, batch_size):
	"""Replaces Keras' native ImageDataGenerator."""
	i = 0
	file_list = os.listdir(directory)
	while True:
		image_batch = []
		for b in range(batch_size):
			if i == len(file_list):
				i = 0

				combined = list(zip(file_list, labels))
				random.shuffle(file_list)
			sample = file_list[i]
			i += 1
			image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
			image_batch.append((image.astype(float) - 128) / 128)

		yield np.array(image_batch)


def read_image_list():
	DATA_DIR = expanduser("~") + '/Documents/Projects/Datasets/ISIC2018/'
	TASK3_DATA_DIR = DATA_DIR + 'ISIC2018_Task3_Training_Input/'
	TASK3_TRUTH_FILE = DATA_DIR + 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

	pattern = re.compile(r"\w+\.jpg")

	image_files = []
	truth_file = open(TASK3_TRUTH_FILE)
	csv_reader = csv.reader(truth_file)
	for row in csv_reader:
		image_files.append(TASK3_DATA_DIR + row[0] + '.jpg')
	image_files = image_files[1:]

	labels = genfromtxt(TASK3_TRUTH_FILE, delimiter=',')
	labels = labels[1:, 1:]

	# Uncomment for index instead of one-hot
	# labels = [np.where(r==1)[0][0] for r in labels]

	return image_files, labels 

def load_images(image_list, img_rows, img_cols):
	images = np.empty((len(image_list), img_rows, img_cols, 3))
	i = 0
	for f in image_list:
		image = cv2.imread(image_list[i])
		image = cv2.resize(image, (img_rows, img_cols)) 
		images[i, :, :, :] = image
		i += 1
		
	# Switch RGB to BGR order
	images = images[:, :, :, ::-1]

	# Subtract ImageNet mean pixel
	images[:, :, :, 0] -= 103.939
	images[:, :, :, 1] -= 116.779
	images[:, :, :, 2] -= 123.68

	if K.image_dim_ordering() == 'th':
		images = np.moveaxis(images, -1, 1)

	return images

def divide_images(X_all, Y_all, validation_size):

	train_index_all = []
	valid_index_all = []

	for c in range(Y_all.shape[1]):
		class_indices = np.where(Y_all[:, c] == 1)[0]
		class_size = class_indices.shape[0]

		train_indices = class_indices[: int(round(class_size * (1-validation_size)))].tolist()
		valid_indices = class_indices[int(round(class_size * (1-validation_size))) :].tolist()

		train_index_all.extend(train_indices)
		valid_index_all.extend(valid_indices)
	
	X_train = X_all[train_index_all, :, :, :]
	Y_train = Y_all[train_index_all, :]

	X_valid = X_all[valid_index_all, :, :, :]
	Y_valid = Y_all[valid_index_all, :]

	return X_train, Y_train, X_valid, Y_valid

def load_ISIC2018(img_rows, img_cols, validation_size):
	image_list, Y_all = read_image_list()
	X_all = load_images(image_list, img_rows, img_cols)
	X_train, Y_train, X_valid, Y_valid = divide_images(X_all, Y_all, validation_size)

	return X_train, Y_train, X_valid, Y_valid

if __name__ == '__main__':
	K.set_image_dim_ordering('th')

	img_rows, img_cols = 299, 299 # Resolution of inputs
	channel = 3
	num_classes = 7
	batch_size = 1
	nb_epoch = 90
	validation_size = 0.2

	# Example to fine-tune on 3000 samples from Cifar10
	# # Load Cifar10 data. Please implement your own load_data() module for your own dataset
	# X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

	# Load our model
	model = inception_v3_model(img_rows, img_cols, channel, num_classes)
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	
	# Create data import pipeline
	X_train, Y_train, X_valid, Y_valid = load_ISIC2018(img_rows, img_cols, validation_size)

	# Data augmentation
	train_datagen = ImageDataGenerator(
        rotation_range=179,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
				channel_shift_range=0.2,
        fill_mode='nearest')

	valid_datagen = ImageDataGenerator(
        rotation_range=179,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
				channel_shift_range=0.2,
        fill_mode='nearest')

	# Compute quantities required for featurewise normalization
	train_datagen.fit(X_train)
	train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

	valid_datagen.fit(X_valid)
	validation_generator = valid_datagen.flow(X_valid, Y_valid, batch_size=batch_size)

	# Setup checkpoints
	filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	# Fits the model on batches with real-time data augmentation
	# model.fit_generator(train_generator,
  #                   steps_per_epoch=X_train.shape[0] // batch_size, epochs=nb_epoch, 
	# 									validation_data=validation_generator, validation_steps=X_valid.shape[0] // batch_size, callbacks=callbacks_list)
	# model.save_weights('final_inceptionV3.h5')

	# # Start fine tuning
	# model.fit(X_train, Y_train,
	# 		  batch_size=batch_size,
	# 		  nb_epoch=nb_epoch,
	# 		  shuffle=True,
	# 		  verbose=1,
	# 		  validation_data=(X_valid, Y_valid),
	# 			callbacks=callbacks_list
	# 		  )

	# Make predictions
	predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

	# Cross-entropy loss score
	score = log_loss(Y_valid, predictions_valid)


# images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
# labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

# input_queue = tf.train.slice_input_producer([images, labels],
# 																					num_epochs=nb_epoch,
# 																					shuffle=True)

# image, label = read_images_from_disk(input_queue)

# # Preprocess images
# image = preprocess_image(image)
# label = preprocess_label(label)

# Start Fine-tuning
# for e in range(nb_epoch)
# 	print("Epoch %d" % e)
# 	model.train(X_batch, Y_batch)


# # Reads pfathes of images together with their labels
# image_list, label_list = read_image_list()

# images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
# labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

# # Makes an input queue
# input_queue = tf.train.slice_input_producer([images, labels],
#                                             num_epochs=nb_epoch,
#                                             shuffle=True)

# image, label = read_images_from_disk(input_queue)

# # Optional Preprocessing or Data Augmentation
# # tf.image implements most of the standard image augmentation
# image = preprocess_image(image)
# label = preprocess_label(label)

# # Optional Image and Label Batching
# image_batch, label_batch = tf.train.batch([image, label],
#                                           batch_size=batch_size)
