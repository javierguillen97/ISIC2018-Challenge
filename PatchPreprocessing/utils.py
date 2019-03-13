import numpy as np
import cv2
import os

def PaddingCrop(image, A=224, b=20):
    h, w = image.shape[:2]
    d = A-2*b
    assert d > 0, 'Window size (A-2*b) must be positive'
    padding = ((b, b + (d - h)%d), (b, b + (d - w)%d), (0, 0))
    image = np.pad(image, padding, mode='constant')
    mx = int(np.ceil(h / d))
    my = int(np.ceil(w / d))
    return np.stack([image[kx*d:kx*d + A, ky*d:ky*d + A] for kx in range(mx) for ky in range(my)], axis=0).reshape(mx, my, A, A, -1)

def Merge(image_array, A=224, b=20):
    return np.vstack([np.hstack([h[b:A-b, b:A-b] for h in v]) for v in image_array])

def main():
    b = 20
    image = cv2.imread('/scratch/group/xqian-group/ISIC2018/ISIC2018_Task1-2_Training_Input/ISIC_0014658.jpg')
    cv2.imwrite('../../results/original.jpg', image)
    img_arr = PaddingCrop(image, b=b)
    [cv2.imwrite('../../results/paddingcrop_{}x{}_{}by{}.jpg'.format(image.shape[0], image.shape[1], kx, ky), img_arr[kx, ky]) for kx in range(img_arr.shape[0]) for ky in range(img_arr.shape[1])]
    img_merge = Merge(img_arr, b=b)[:image.shape[0], :image.shape[1]]
    cv2.imwrite('../../results/merge.jpg', img_merge)
    original_shape = None
    original_dtype = object
    img_dict = {}
    for f in os.listdir('../../results/'):
        if f.endswith('.jpg') and 'paddingcrop' in f:
            img_name = f.split('.')[0]
            print(img_name)
            original_shape = [int(k) for k in img_name.split('_')[1].split('x')]
            idx, idy = [int(k) for k in img_name.split('_')[2].split('by')]
            img_dict[(idx, idy)] = cv2.imread(os.path.join('../../results/', f))
            print(img_dict[(idx, idy)])
            original_dtype = img_dict[(idx, idy)].dtype
    ids = np.array(list(img_dict.keys()))
    dims = ids.max(0) + 1
    img_arr2 = np.zeros(dims, dtype=object)
    for k, v in img_dict.items():
        img_arr2[k] = v
    img_arr2 = np.stack([np.stack([k for k in v]) for v in img_arr2]).astype(original_dtype)
    print(img_arr2.dtype)
    print(img_arr2.shape)
    print(original_dtype)
    img_merge2 = Merge(img_arr2, b=b)[:original_shape[0], :original_shape[1]]
    cv2.imwrite('../../results/merge2.jpg', img_merge2)

if __name__ == '__main__':
    main()
