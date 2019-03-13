import os

import cv2

from utils import PaddingCrop

from tqdm import tqdm

def main():
    image_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task1-2_Training_Input/'
    mask_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task1_Training_GroundTruth/'

    image_paddingcrop_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task1-2_Training_Padding_Crop/Training_Input_Padding_Crop/'
    mask_paddingcrop_path = '/scratch/group/xqian-group/ISIC2018/ISIC2018_Task1-2_Training_Padding_Crop/Task1_Training_GroundTruth_Padding_Crop/'
    if not os.path.exists(image_paddingcrop_path):
        os.makedirs(image_paddingcrop_path)
    if not os.path.exists(mask_paddingcrop_path):
        os.makedirs(mask_paddingcrop_path)

    A = 224
    b = 20
    
    for f in tqdm(os.listdir(image_path)):
        if f.endswith('.jpg'):
            img_name = f.split('.')[0]
            im = cv2.imread(os.path.join(image_path, f))
            im_shape = im.shape
            im_arr = PaddingCrop(im, A=A, b=b)
            [cv2.imwrite(os.path.join(image_paddingcrop_path, '{}_{}x{}_{}by{}.jpg'.format(img_name, im_shape[0], im_shape[1], kx, ky)), im_arr[kx, ky]) for kx in range(im_arr.shape[0]) for ky in range(im_arr.shape[1])]
    print('Done for images.')
    
    for f in tqdm(os.listdir(mask_path)):
        if f.endswith('.png'):
            msk_name = f.split('.')[0]
            mk = cv2.imread(os.path.join(mask_path, f))
            mk_arr = PaddingCrop(mk, A=A, b=b)
            [cv2.imwrite(os.path.join(mask_paddingcrop_path, '{}_{}x{}_{}by{}.png'.format(msk_name, mk.shape[0], mk.shape[1], kx, ky)), mk_arr[kx, ky]) for kx in range(mk_arr.shape[0]) for ky in range(mk_arr.shape[1])]
    print('Done for masks.')

    print('Done.')

if __name__ == '__main__':
    main()
