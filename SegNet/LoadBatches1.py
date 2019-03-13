import numpy as np
import cv2
import glob
import itertools

def getImageArr(path, width, height, overlap_percentage, imgNorm="sub_mean", odering='channels_first'):
  try:
    img = cv2.imread(path, 1)

    if imgNorm == "sub_and_divide":
      img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    elif imgNorm == "sub_mean":
      # img = cv2.resize(img, ( width , height ))
      img = img.astype(np.float32)
      img[:,:,0] -= 103.939
      img[:,:,1] -= 116.779
      img[:,:,2] -= 123.68
    elif imgNorm == "divide":
      # img = cv2.resize(img, ( width , height ))
      img = img.astype(np.float32)
      img = img/255.0

    image_patches = divideIntoPatches(img, width, height, overlap_percentage)
    if odering == 'channels_first':
      i = 0
      for img in image_patches:
        img = np.rollaxis(img, 2, 0)
        image_patches[i] = img
        i += 1

    return image_patches
  except Exception as e:
    print(path, e)
    img = np.zeros((height, width, 3))
    if odering == 'channels_first':
      img = np.rollaxis(img, 2, 0)

    image_patches = divideIntoPatches(img, width, height, overlap_percentage)
    return image_patches

def getSegmentationArr(path, nClasses, input_width, input_height, width , height, overlap_percentage):
  try:
    image = cv2.imread(path, 1)
    image_patches = divideIntoPatches(image, input_width, input_height, overlap_percentage)

    seg_labels = []
    for patch in image_patches:
      patch = cv2.resize(patch, (width, height))
      patch = patch[:, : , 0]

      seg_label = np.zeros((height, width, nClasses))
      seg_label[:, :, 0] = (patch == 0).astype(int)
      seg_label[:, :, 1] = (patch == 255).astype(int)
      seg_label = np.reshape(seg_label, (width*height, nClasses))

      # img = img[:, : , 0]
      # for c in range(nClasses):
      # seg_label[: , : , c ] = (img == c ).astype(int)

      seg_labels.append(seg_label)

  except Exception as e:
    print(e)
  return seg_labels

def divideIntoPatches(image, img_cols, img_rows, overlap_percentage):
  images = []

  current_height = 0
  while current_height < image.shape[0] - img_rows:
    current_width = 0
    while current_width < image.shape[1] - img_cols:
      image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
      images.append(image_section)
      current_width += int(img_cols * (1-overlap_percentage))
      if current_width > image.shape[1] - img_cols:
        current_width = image.shape[1] - img_cols
    # Last block
    image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
    images.append(image_section)
    current_height += int(img_rows * (1-overlap_percentage))
    if current_height > image.shape[0] - img_rows:
      current_height = image.shape[0] - img_rows

  # Last row
  current_width = 0
  while current_width < image.shape[1] - img_cols:
    image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
    images.append(image_section)
    current_width += int(img_cols * (1-overlap_percentage))
    if current_width > image.shape[1] - img_cols:
      current_width = image.shape[1] - img_cols
    image_section = image[current_height : current_height + img_rows, current_width : current_width + img_cols, :]
    images.append(image_section)

  return images

def imageSegmentationGenerator(images_path, segs_path,  batch_size,  n_classes, input_height, input_width, output_height, output_width, overlap_percentage):
  
  assert images_path[-1] == '/'
  assert segs_path[-1] == '/'

  images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") +  glob.glob(images_path + "*.jpeg")
  images.sort()
  segmentations  = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") +  glob.glob(segs_path + "*.jpeg")
  segmentations.sort()

  assert len( images ) == len(segmentations)
  for im , seg in zip(images,segmentations):
    # assert(im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    assert(im.split('/')[-1].split(".")[0] + '_segmentation' ==  seg.split('/')[-1].split(".")[0])

  zipped = itertools.cycle(zip(images,segmentations))

  im_patches = []
  seg_patches = []
  while True:
    X = []
    Y = []

    i = 0
    while i < batch_size: 
      if not im_patches or not seg_patches :
        im, seg = next(zipped)
        im_patches = getImageArr(im, input_width, input_height, overlap_percentage)
        seg_patches = getSegmentationArr(seg, n_classes, input_width, input_height, output_width, output_height, overlap_percentage)
      X.append(im_patches.pop(0))
      Y.append(seg_patches.pop(0))
      i += 1
    yield np.array(X) , np.array(Y)

# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )
