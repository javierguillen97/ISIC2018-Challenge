"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = '../../logs/'

# Results directory
RESULTS_DIR = '../../results/'

############################################################
#  Image IDs Splitting
############################################################
import random
from random import shuffle

############################################################
#  Configurations
############################################################

class LesionSegConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lesion_crop"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + lesion

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Don't exclude based on confidence since only two classes
    DETECTION_MIN_CONFIDENCE = 0

    BACKBONE = 'resnet50'

############################################################
#  Dataset
############################################################

class LesionSegDataset(utils.Dataset):

    def load_lesion(self, image_dir, subset, mask_dir=None, mask_name_list=None):
        """Load a subset of the Lesion dataset.
        image_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("lesion", 1, "lesion")
        self.add_class("attribute_globules", 2, "attribute_globules")
        self.add_class("attribute_milia_like_cyst", 3, "attribute_milia_like_cyst")
        self.add_class("attribute_negative_network", 4, "attribute_negative_network")
        self.add_class("attribute_pigment_network", 5, "attribute_pigment_network")
        self.add_class("attribute_streaks", 6, "attribute_streaks")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        
        img_id_list = []
        for f in os.listdir(image_dir):
            if f.endswith('.jpg'):
                img_id_list.append(f.split('.')[0])
        random.seed(42)
        test_size = 0.2
        shuffle(img_id_list)
        if subset == 'train':
            img_ids = img_id_list[int(len(img_id_list)*test_size):]
        if subset == 'val':
            img_ids = img_id_list[:int(len(img_id_list)*test_size)]

        # Add images
        for img_id in img_ids:
            image_path = os.path.join(image_dir, img_id + '.jpg')
            #image = skimage.io.imread(image_path)

            self.add_image(
                "lesion",
                image_id=img_id,  # use file name as a unique image id
                path=image_path,
                maskdir=mask_dir,
                masknames=mask_name_list)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a lesion dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "lesion":
            return super(self.__class__, self).load_mask(image_id)
        mask_dir = image_info["maskdir"]
        mask_names = image_info["masknames"]

        # Read mask files from .png image
        mask = []
        mask_id = []
        curr_id = 0
        for mask_name in mask_names:
            mask_file = os.path.join(mask_dir, '{}_{}.png'.format(image_info["id"], mask_name))
            mask.append(skimage.io.imread(mask_file).astype(np.bool))

            mask_id.append(curr_id+1)
            curr_id = (curr_id+1) % len(mask_names)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        
        return mask, np.array(mask_id, dtype=np.int32)
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lesion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model, image_dir, mask_dir, mask_name_list, layers):
    assert layers in ['heads', 'all']
    """Train the model."""
    # Training dataset.
    dataset_train = LesionSegDataset()
    dataset_train.load_lesion(image_dir, "train", mask_dir, mask_name_list)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LesionSegDataset()
    dataset_val.load_lesion(image_dir, "val", mask_dir, mask_name_list)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')
    if layers == 'all':
        print("Training all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100,
                    layers='all')

def mask_gather(mask):
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        result = np.any(mask, axis=-1)*255
        result = result.astype(np.uint8)
    else:
        result = None
    return result
    
def detect(model, image_dir, subset):
    assert subset in ['train', 'val']
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = 'submit_{:%Y%m%dT%H%M%S}_{}'.format(datetime.datetime.now(), subset)
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = LesionSegDataset()
    dataset.load_lesion(image_dir, subset, mask_dir=None, mask_name_list=None)
    dataset.prepare()

    # Load over images
    for image_id in dataset.image_ids:
        # Run model detection and generate the color splash effect
        print('Running on {}'.format(image_id))
        # Load image
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Get mask
        mask = mask_gather(r['masks'])
        # Save mask
        file_name = '{}_lesion_segmentation.png'.format(image_id)
        file_path = os.path.join(submit_dir, file_name)
        skimage.io.imsave(file_path, mask)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lesion segmentation.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--imagedir', required=False,
                        metavar="/path/to/lesion/imagedir/",
                        help='Directory of the LesionSeg images')
    parser.add_argument('--maskdir', required=False,
                        metavar="/path/to/lesion/maskdir/",
                        help='Directory of the LesionSeg masks')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Data subset",
                        help='Subset of dataset to run prediction on')
    parser.add_argument('--layers', required=False,
                        default='heads',
                        metavar='Layers to train',
                        help="Layers of the model to train, 'heads'(default) or 'all'")
    args = parser.parse_args()

    # Validate arguments
    assert args.imagedir, "Argument --imagedir is required"
    assert args.maskdir, "Argument --maskdir is required"
    if args.command == "detect":
        assert args.subset, "Argument --subset is required"

    print("Weights: ", args.weights)
    print("Imagedir: ", args.imagedir)
    print("Maskdir: ", args.maskdir)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    masknames_used = ['segmentation', 'attribute_globules', 'attribute_milia_like_cyst', 
                    'attribute_negative_network', 'attribute_pigment_network', 'attribute_streaks']

    # masknames_used = ['segmentation']

    # Configurations
    if args.command == "train":
        config = LesionSegConfig()
    else:
        class InferenceConfig(LesionSegConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            BACKBONE = 'resnet50'
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.imagedir, args.maskdir, masknames_used, args.layers)
    elif args.command == "detect":
        detect(model, args.imagedir, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
