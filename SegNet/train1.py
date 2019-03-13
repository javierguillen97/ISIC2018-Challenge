import LoadBatches1
import FCN32
import FCN8
import VGGSegnet
import VGGUnet
import glob
import pickle

from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def jaccard_distance(y_true, y_pred, smooth=100):
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


# train_images_path = "dataset1/images_prepped_train/"
# train_segs_path = "dataset1/annotations_prepped_train/"
train_images_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1-2_Training_Input/"
train_segs_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth/"

train_batch_size = 3

# n_classes = 10
n_classes = 2

input_height = 224
input_width = 224
overlap_percentage = 0.1

# val_images_path = "dataset1/images_prepped_test/"
# val_segs_path = "dataset1/annotations_prepped_test/"
val_images_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1-2_Validation_Input/"
val_segs_path = "/Users/Ardywibowo/Documents/Projects/Datasets/ISIC2018/ISIC2018_Task1_Validation_GroundTruth/"

val_batch_size = 3

load_weights = "data/vgg19_weights_th_dim_ordering_th_kernels.h5"
model_name = "vgg_segnet"

save_weights_path = "weights/isic_task1/"
epochs = 100

rmsprop = RMSprop(lr=0.001, decay=1e-6, epsilon=0.1)

modelFns = {'vgg_segnet':VGGSegnet.VGGSegnet, 'vgg_unet':VGGUnet.VGGUnet, 'vgg_unet2':VGGUnet.VGGUnet2, 'fcn8':FCN8.FCN8, 'fcn32':FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
plot_model(m, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
m.compile(loss=['categorical_crossentropy'],
      optimizer=rmsprop,
      metrics=['accuracy'])

# m.load_weights(load_weights)

# print "Model output shape" ,  m.output_shape
output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches1.imageSegmentationGenerator(train_images_path, train_segs_path,  train_batch_size, n_classes, input_height, input_width, output_height, output_width, overlap_percentage)
G2  = LoadBatches1.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height, input_width, output_height, output_width, overlap_percentage)

filepath = save_weights_path + "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

train_files = glob.glob(train_images_path + "*.jpg") + glob.glob(train_images_path + "*.png") + glob.glob(train_images_path + "*.jpeg")
valid_files = glob.glob(val_images_path + "*.jpg") + glob.glob(val_images_path + "*.png") + glob.glob(val_images_path + "*.jpeg")

train_length = len(train_files) * 100
valid_length = len(valid_files) * 100

# history = m.fit_generator(G, steps_per_epoch=train_length // train_batch_size, epochs=epochs, 
# 		validation_data=G2, validation_steps=valid_length // val_batch_size, callbacks=callbacks_list)
# m.save_weights(save_weights_path + 'final_segnet.h5')

history = m.fit_generator(G, steps_per_epoch=100, epochs=100, 
		validation_data=G2, validation_steps=1, callbacks=callbacks_list)
m.save_weights(save_weights_path + 'final_segnet.h5')

# for ep in range(epochs):
# 	m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
# 	m.save_weights(save_weights_path + "." + str(ep))
# 	m.save(save_weights_path + ".model." + str(ep))

f = open('history.pckl', 'wb')
pickle.dump(history, f)
f.close()
