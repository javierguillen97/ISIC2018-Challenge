import cv2
import csv
import numpy as np
from os.path import expanduser
import re

def read_image_list():
	DATA_DIR = expanduser("~") + '/Documents/Projects/Datasets/ISIC2018/'
	TASK3_DATA_DIR = DATA_DIR + 'ISIC2018_Task3_Training_Input/'
	TASK3_TRUTH_FILE = DATA_DIR + 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

	pattern = re.compile(r"\w+\.(jpg|png)")

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

def load_images_folder(folder_name, img_rows, img_cols, overlap_percentage)
	pattern = re.compile(r"\w+\.(jpg|png)")
	image_list = [f for f in listdir(folder_name) if (isfile(join(folder_name, f)) & bool(re.match(pattern, f)))]
	images = np.empty((0, img_rows, img_cols, 3))

	i = 0
	for f in image_list:
		image = cv2.imread(folder_name + image_list[i])
		# Divide images into blocks of (img_rows, img_cols)
		current_height = 0
		while current_height < image.shape[0] - img_rows:
			current_width = 0
			while current_width < image.shape[1] - img_cols:
				image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
				image_section = np.expand_dims(image_section, axis=0)
				images = np.concatenate((images, image_section), axis=0)
				current_width += int(img_cols * (1-overlap_percentage))
				if current_width > image.shape[1] - img_cols:
					current_width = image.shape[1] - img_cols
			# Last block
			image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
			image_section = np.expand_dims(image_section, axis=0)
			images = np.concatenate((images, image_section), axis=0)
			current_height += int(img_rows * (1-overlap_percentage))
			if current_height > image.shape[0] - img_rows:
				current_height = image.shape[0] - img_rows

		# Last row
		current_width = 0
		while current_width < image.shape[1] - img_cols:
			image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
			image_section = np.expand_dims(image_section, axis=0)
			images = np.concatenate((images, image_section), axis=0)
			current_width += int(img_cols * (1-overlap_percentage))
			if current_width > image.shape[1] - img_cols:
				current_width = image.shape[1] - img_cols
			image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
			image_section = np.expand_dims(image_section, axis=0)
			images = np.concatenate((images, image_section), axis=0)

		# image = cv2.resize(image, (img_rows, img_cols)) 
		images[i, :, :, :] = image
		i += 1
	
	# Switch RGB to BGR order
	images = images[:, :, :, ::-1]

	return images
