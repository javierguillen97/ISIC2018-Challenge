# unet
from os.path import expanduser
import tensorflow as tf
import numpy as np
import model

import TensorflowUtils as utils
import datetime
import random
import pdb

# training loss
def train(loss_val, var_list, LearningRate, Debug):
    optimizer = tf.train.AdamOptimizer(LearningRate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if Debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)
    # return optimizer

# main function
# def main(CHANNEL, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, ImageData, NoisyData, CleanData, 
# 	     MAX_EPOCH, BatchSize, KEEP_PROB, REG_WEIGHT, LearningRate, RESTORE, DirSave, DirLoad = None, Debug = True):
def main(CHANNEL, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, NoisyInput, NoisyOutput, CleanInput, CleanOutput, 
	MAX_EPOCH, BatchSize, KEEP_PROB, REG_WEIGHT, LearningRate, RESTORE, SizeLimitation, DirSave, DirLoad = None, Debug = True):
    
	ImageSize = [NoisyInput.shape[1], NoisyInput.shape[2]]
	OutputImageSize = [NoisyOutput.shape[1], NoisyOutput.shape[2]]
	NAME = 'unet'
	tf.reset_default_graph()
	with tf.name_scope("Input"):
		image = tf.placeholder(tf.float32, shape=[None, ImageSize[0], ImageSize[1], CHANNEL], name="input_image")
		NoisyLabel = tf.placeholder(tf.int64, shape=[None, OutputImageSize[0], OutputImageSize[1], NClass], name = "NoisyLabel")
		CleanLabel = tf.placeholder(tf.int64, shape=[None, OutputImageSize[0], OutputImageSize[1], NClass], name = "CleanLabel")
		keep_prob = tf.placeholder(tf.float32, shape=[], name = "keep_prob")
		utils.add_to_image_summary(image)
		
		# 50 data are split to training set and validatation set
		# 70% training set and 30% validation set

		# if np.max(image_data) > 1:
		# 	# image_data = image_data
		# 	image_data = model.normalize(image_data)


		# size = np.shape(NoisyInput)[0]
		# tr_size = np.int(size * 0.7)
		# BATCHES = tr_size
		# val_size = size - tr_size

		# tr_image_data = ImageData[0 : tr_size]
		# tr_label_data = NoisyData[0 : tr_size]
		# TrCleanLabel = CleanData[0 : tr_size]

		# val_image_data = ImageData[tr_size : size]
		# val_label_data = NoisyData[tr_size : size]
		# ValCleanLabel = CleanData[tr_size : size]

		BATCHES = np.shape(NoisyInput)[0]

		tr_image_data = NoisyInput
		tr_label_data = NoisyOutput

		val_image_data = CleanInput
		val_label_data = CleanOutput	

	with tf.name_scope("net"):

		if RESTORE == True:
			weights = np.load(DirLoad)
			NoisyLoss, CleanOut, variables,dw_h_convs = model.create_unet(image, NoisyLabel, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, 
   														 	   keep_prob, NAME, REG_WEIGHT, Debug, restore = RESTORE, weights = weights)

    		
			print("Model restored...")
		else:
			NoisyLoss, CleanOut, variables, dw_h_convs = model.create_unet(image, NoisyLabel, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, keep_prob, NAME, REG_WEIGHT, Debug)

 		CleanLoss = utils.cross_entropy(tf.cast(tf.reshape(CleanLabel, [-1, NClass]), tf.float32), 
								tf.reshape(CleanOut, [-1, NClass]),'Cleanloss')

		# utils.add_scalar_summary(CleanLoss)

		# NoiseAcc = tf.reduce_mean(tf.cast(tf.reshape(tf.equal(NoisyLabel, tf.argmax(CleanOut, 3)), [-1]), tf.float32), name = 'NoiseAcc')
		NoiseAcc = tf.reduce_mean(tf.cast(tf.reshape(tf.equal(tf.argmax(NoisyLabel, 3), tf.argmax(CleanOut, 3)), [-1]), tf.float32), name = 'NoiseAcc')
		CleanAcc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(CleanLabel, 3), tf.argmax(CleanOut, 3)), tf.float32), name = 'CleanAcc')

		utils.add_scalar_summary(NoiseAcc)
		utils.add_scalar_summary(CleanAcc)
		utils.add_scalar_summary(CleanLoss)


	with tf.name_scope("Train"):

		trainable_var = tf.trainable_variables()
		train_op = train(NoisyLoss, trainable_var, LearningRate, Debug)
		print("Setting up summary op...")
		
		summary_op = tf.summary.merge_all()

		# uncomment BELOW TO RUNNING ON CPU
		# pdb.set_trace()
		# config = tf.ConfigProto(device_count = {'GPU': 0})
		# sess = tf.Session(config=config)	
		# uncomment to run on GPU
		sess = tf.Session()
		###############################

		print("Setting up Saver...")
		saver = tf.train.Saver()
		summary_writer = tf.summary.FileWriter(DirSave, sess.graph)

		#################
		# Insert code of data file checking here
		#################

		sess.run(tf.global_variables_initializer())
		tr_image_batch1 = tr_image_data[0  : SizeLimitation]
		tr_label_batch1 = tr_label_data[0  : SizeLimitation]

		val_image_batch = val_image_data[0 : SizeLimitation]
		val_label_batch = val_label_data[0 : SizeLimitation]

		total_iter = 0
		for epoch in range (MAX_EPOCH):
			for batch in range(0, BATCHES / BatchSize):
			# for batch in [0]:
		    	# image: [batch, row, col, channel]
		   		# label: [batch, row, col, n_class]
				tr_image_batch = tr_image_data[batch * BatchSize : batch * BatchSize + BatchSize]
				tr_label_batch = tr_label_data[batch * BatchSize : batch * BatchSize + BatchSize]


				tr_feed_dict = {image: tr_image_batch, NoisyLabel: tr_label_batch,  keep_prob: np.float32(KEEP_PROB)}
				tr_feed_dict1 = {image: tr_image_batch1, NoisyLabel: tr_label_batch1, CleanLabel:val_label_batch, keep_prob: np.float32(KEEP_PROB)}
				val_feed_dict = {image: val_image_batch, CleanLabel: val_label_batch, keep_prob: np.float32(KEEP_PROB)}
				# pdb.set_trace()
				
		        # trainining set
				if (total_iter) % 10 == 0:
	
					# pre_seg, _NoisyLoss, _CleanLoss, _CleanAcc, _NoiseAcc, tr_variables, summary_str = sess.run([CleanOut, NoisyLoss, CleanLoss, CleanAcc, NoiseAcc, 
					#																							   variables, summary_op], feed_dict = tr_feed_dict1)
					_dw_h_convs, _NoisyLoss, pre_seg, _NoiseAcc, tr_variables, summary_str,  = sess.run([dw_h_convs, NoisyLoss, CleanOut, NoiseAcc, variables, summary_op], feed_dict = tr_feed_dict1)

					
					# print("Iter: %d, TrainNoisyLoss: %g, TrainNoiseAcc: %g" % (total_iter, _NoisyLoss, _NoiseAcc))      
					summary_writer.add_summary(summary_str, total_iter)
					saver.save(sess, DirSave + "model.ckpt", total_iter)
					np.save(DirSave + "weights", tr_variables)
			
				# validation set
				if (total_iter) % 50 == 0:
					# _NoisyLoss, _CleanLoss, _NoiseAcc, _CleanAcc = sess.run([NoisyLoss, CleanLoss, NoiseAcc, CleanAcc], feed_dict = val_feed_dict)
					# print("Iter: %d, ValNoisyLoss: %g, ValCleanLoss: %g, ValNoiseAcc: %g, ValCleanAcc: %g, curent_time: %s" % 
					# 	  (total_iter, _NoisyLoss, _CleanLoss, _NoiseAcc, _CleanAcc, str(datetime.datetime.now())))              

					_CleanLoss, _CleanAcc = sess.run([CleanLoss, CleanAcc], feed_dict = val_feed_dict)
					# print("Iter: %d, ValCleanLoss: %g, ValCleanAcc: %g, curent_time: %s" % 
					# 	  (total_iter, _CleanLoss, _CleanAcc, str(datetime.datetime.now())))     
				sess.run(train_op, feed_dict=tr_feed_dict)
				total_iter += 1    

			new_index = random.sample(range(BATCHES), BATCHES)
			tr_image_data = tr_image_data[new_index]
			tr_label_data = tr_label_data[new_index]

	sess.close()
	return _CleanAcc, _NoiseAcc
def Predict(CHANNEL, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, KEEP_PROB, ImageData, NoisyData, CleanData, Run, NoiseLevel, DirLoad):
	ImageSize = [ImageData.shape[1], ImageData.shape[2]]
	NAME = 'unet'
	tf.reset_default_graph()

	with tf.name_scope("Input"):
		image = tf.placeholder(tf.float32, shape=[None, ImageSize[0], ImageSize[1], CHANNEL], name="input_image")
		NoisyLabel = tf.placeholder(tf.int64, shape=[None, ImageSize[0], ImageSize[1], NClass], name = "NoisyLabel")
		CleanLabel = tf.placeholder(tf.int64, shape=[None, ImageSize[0], ImageSize[1], NClass], name = "CleanLabel")
		keep_prob = tf.placeholder(tf.float32, shape=[], name = "keep_prob")

		# 50 data are split to training set and validatation set
		# 70% training set and 30% validation set

		# if np.max(image_data) > 1:
		# 	# image_data = image_data
		# 	image_data = model.normalize(image_data)




	with tf.name_scope("net"):

		weights = np.load(DirLoad)
		NoisyLoss, CleanOut, variables = model.create_unet(image, NoisyLabel, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, 
														   keep_prob, NAME, restore = True, weights = weights)

		
		print("Model restored...")

		NoiseAcc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(NoisyLabel, 3), tf.argmax(CleanOut, 3)), tf.float32), name = 'NoiseAcc')
		CleanAcc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(CleanLabel, 3), tf.argmax(CleanOut, 3)), tf.float32), name = 'CleanAcc')

	with tf.name_scope("Predict"):
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		feed_dict = {image: ImageData, NoisyLabel: NoisyData, CleanLabel:CleanData, keep_prob: np.float32(KEEP_PROB)}
		_CleanOut, _NoiseAcc, _CleanAcc = sess.run([CleanOut, NoiseAcc, CleanAcc], feed_dict = feed_dict)
		print("TrainNoiseError: %g, TrainCleanError: %g, Run: %d, Flipping Prob: %g" % (1 - _NoiseAcc, 1 - _CleanAcc, Run, NoiseLevel))   

	return _CleanOut

def PredictAll(CHANNEL, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, KEEP_PROB, ImageData, LabelData, DirLoad):
	ImageSize = [ImageData.shape[1], ImageData.shape[2]]
	NumOfImage = ImageData.shape[0]
	NAME = 'unet'
	tf.reset_default_graph()

	with tf.name_scope("Input"):
		image = tf.placeholder(tf.float32, shape=[None, ImageSize[0], ImageSize[1], CHANNEL], name="input_image")
		LabelDataPH = tf.placeholder(tf.int64, shape=[None, ImageSize[0], ImageSize[1], NClass], name = "LabelData")
		keep_prob = tf.placeholder(tf.float32, shape=[], name = "keep_prob")

		# 50 data are split to training set and validatation set
		# 70% training set and 30% validation set

		# if np.max(image_data) > 1:
		# 	# image_data = image_data
		# 	image_data = model.normalize(image_data)



	with tf.name_scope("net"):

		weights = np.load(DirLoad)
	
		Loss, CleanOut, variables, dw_h_convs = model.create_unet(image, LabelDataPH, NClass, FILTER_SIZE, NUM_OF_FEATURE, NUM_OF_LAYERS, 
														   		  keep_prob, NAME, restore = True, weights = weights)
	
		print("Model restored...")
		
		Acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(LabelDataPH, 3), tf.argmax(CleanOut, 3)), tf.float32), name = 'Acc')

	with tf.name_scope("Predict"):
		AllAcc = []
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for i in range(NumOfImage):
			feed_dict = {image: ImageData[i : i + 1], LabelDataPH: LabelData[i : i + 1], keep_prob: np.float32(KEEP_PROB)}
			_CleanOut, _Acc = sess.run([CleanOut, Acc], feed_dict = feed_dict)
			AllAcc.append(_Acc)
		
		AvgErr = 1 - np.mean(AllAcc)
		print("Error: %g" % AvgErr)   

	return AvgErr

		


