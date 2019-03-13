from __future__ import division
import os
import cv2
from os.path import expanduser
import numpy as np


DATA_DIR = expanduser("~") + '/PycharmProjects/test0612/datasets/ISIC2018/data/'
TASK3_DATA_DIR = DATA_DIR + 'ISIC2018_Task3_Training_Input/'


filepath_images=TASK3_DATA_DIR



filename_list=os.listdir(filepath_images)



def load_images(filename_list):
     train_set_x = []
     ids_x = []
     cropx=75
     cropy=25
     print('-' * 50)
     print('Loading all images from {0}...'.format(filepath_images))
     print('-' * 50)

     for filename in filename_list:
       if filename.endswith('jpg'):
        img1=cv2.imread(filepath_images+filename)
        ids_x.append(filename)
        if img1 is None:
             continue


        res1=img1[:,cropx:img1.shape[1]-cropx,:]
        #res1=cv2.resize(img1,(249,249))
        cv2.imwrite('data/resizedimage/resized--%s'%(filename), res1)
        train_set_x.append(res1)
        X2 = color_constancy(res1, power=6, gamma=None)
        cv2.imwrite('data/colornormimage/colornorm--%s' % (filename), X2)

     print('-' * 50)
     print('All images in {0} loaded.'.format(filepath_images))
     print('-' * 50)
     print (ids_x)
     return train_set_x,ids_x




def write_images_npy():
     """
     Loads the validation data set including X and study ids and saves it to .npy file.
     """
     print('-' * 50)
     print('Writing  data to .npy file...')
     print('-' * 50)

     X,ids_x= load_images(filename_list)
     X = np.array(X, dtype=np.uint8)
     np.save('data/X.npy', X)
     np.save('data/ids_x.npy',ids_x)
     #np.save('data/ids_validate.npy', study_ids)
     print('Done.')





def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X.npy')
    y = np.load('data/y.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.
    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split1 = X.shape[0] * split_ratio
    split1=int(round(split1))
    X_test = X[:split1, :, :, :]
    y_test = y[:split1, :, :, :]

    X_train = X[split1:, :, :, :]
    y_train = y[split1:, :, :, :]

    X_validate=X_train[:split1, :, :, :]
    y_validate=y_train[:split1, :, :, :]


    return X_train, y_train, X_validate, y_validate, X_test, y_test


def color_constancy(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in xrange(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)

"""


#write_images_npy()
write_truth_npy()

X,y=load_train_data()
X_train, y_train, X_validate, y_validate, X_test, y_test=split_data(X, y, split_ratio=0.2)

np.save('data/X_train.npy',X_train)
np.save('data/y_train.npy',y_train)
np.save('data/X_validate.npy',X_validate)
np.save('data/y_validate.npy',y_validate)
np.save('data/X_test.npy',X_test)
np.save('data/y_test.npy',y_test)
"""

X,ids_x=load_images(filename_list)



X=X
