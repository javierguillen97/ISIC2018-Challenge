import glob
import cv2
import numpy as np

ground_truth_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1_Validation_GroundTruth/"
result_path = "/Users/Ardywibowo/Documents/Projects/ISIC-Challenge/result/segnet_task1/"

ground_truth_images = glob.glob(ground_truth_path + "*.jpg") + glob.glob(ground_truth_path + "*.png") +  glob.glob(ground_truth_path + "*.jpeg")
result_images = glob.glob(result_path + "*.jpg") + glob.glob(result_path + "*.png") +  glob.glob(result_path + "*.jpeg")

ground_truth_images.sort()
result_images.sort()

total_jaccard = 0
total_jaccard_thresh = 0
for i in range(len(result_images)):
  result = cv2.imread(result_images[i], 1)
  ground_truth = cv2.imread(ground_truth_images[i], 1)

  result = result[:, : , 0]
  result = (result >= 128 ).astype(int)

  ground_truth = cv2.resize(ground_truth, result.shape)
  ground_truth = ground_truth[:, : , 0]
  ground_truth = (ground_truth >= 128 ).astype(int)

  image_sum = np.add(result, ground_truth)
  image_intersection = np.multiply(result, ground_truth)
  image_union = image_sum - image_intersection

  jaccard = np.sum(image_intersection).astype(np.float) / np.sum(image_union).astype(np.float)

  total_jaccard += jaccard
  if jaccard >= 0.65:
    total_jaccard_thresh += jaccard

total_jaccard = total_jaccard / len(result_images)
total_jaccard_thresh = total_jaccard_thresh / len(result_images)

print("Total Jaccard: ", total_jaccard)
print("Thresholded Jaccard: ", total_jaccard_thresh)
