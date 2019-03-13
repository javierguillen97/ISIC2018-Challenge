import tensorflow as tf 
import TensorflowUtils as utils
import numpy as np
import pdb

def unet_downsample(image, filter_size, num_of_feature, num_of_layers,
					keep_prob, name, debug, restore = False, weights = None):
	channels = image.get_shape().as_list()[-1]
	dw_h_convs = {}
	variables = []
	pools = {}
	in_node = image

	# downsample layer
	layer_id = 0
	weight_id = 0
	for layer in range(0, num_of_layers):
		features = 2**layer*num_of_feature
		
		stddev = 0.02
		w1_name = name + '_layer_' + str(layer_id) + '_w_0'
		w2_name = name + '_layer_' + str(layer_id) + '_w_1'
		b1_name = name + '_layer_' + str(layer_id) + '_b_0'
		b2_name = name + '_layer_' + str(layer_id) + '_b_1'
		relu_name = name + '_layer_' + str(layer_id) + '_feat'
		if layer == 0:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:
			    w1 = utils.weight_variable([filter_size, filter_size, channels, features], stddev, w1_name)
		else:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:			
				w1 = utils.weight_variable([filter_size, filter_size, features//2, features], stddev, w1_name)

		if restore == True:
			w2 = utils.get_variable(weights[weight_id], w2_name)
			weight_id+=1
			b1 = utils.get_variable(weights[weight_id], b1_name)
			weight_id+=1
			b2 = utils.get_variable(weights[weight_id], b2_name)
			weight_id+=1
		else:
			w2 = utils.weight_variable([filter_size, filter_size, features, features], stddev, w2_name)
			b1 = utils.bias_variable([features], b1_name)
			b2 = utils.bias_variable([features], b2_name)

		conv1 = utils.conv2d_basic(in_node, w1, b1, keep_prob)
		tmp_h_conv = tf.nn.relu(conv1)
		conv2 = utils.conv2d_basic(tmp_h_conv, w2, b2, keep_prob)
		
		dw_h_convs[layer] = tf.nn.relu(conv2, relu_name)      

		if layer < num_of_layers-1:
		    pools[layer] = utils.max_pool_2x2(dw_h_convs[layer])
		    in_node = pools[layer]
		 
		
		if debug: 
			utils.add_activation_summary(dw_h_convs[layer])
			utils.add_to_image_summary(utils.get_image_summary(dw_h_convs[layer],  relu_name + '_image'))


		variables.extend((w1, w2, b1, b2))

		layer_id+=1

	return dw_h_convs, variables, layer_id, weight_id

def unet_upsample(image, dw_h_convs, variables, layer_id, weight_id, 
				  filter_size, num_of_feature, num_of_layers, keep_prob, 
				  name, debug, restore = False, weights = None):
	new_variables = []
	in_node = dw_h_convs[num_of_layers - 1]
     # upsample layer
	for layer in range(num_of_layers - 2, -1, -1):
		features = 2 ** (layer + 1) * num_of_feature
		stddev = 0.02

		wd_name = name + '_layer_up' + str(layer_id) + '_w'
		bd_name = name + '_layer_up' + str(layer_id) + '_b'
		w1_name = name + '_layer_up_conv' + str(layer_id) + '_w0'
		w2_name = name + '_layer_up_conv' + str(layer_id) + '_w1'
		b1_name = name + '_layer_up_conv' + str(layer_id) + '_b0'
		b2_name = name + '_layer_up_conv' + str(layer_id) + '_b1'
		relu_name = name + '_layer_up_conv' + str(layer_id) + '_feat'

		# pooling size is 2
		if restore == True:
			wd = utils.get_variable(weights[weight_id], wd_name)
			weight_id += 1
			bd = utils.get_variable(weights[weight_id], bd_name)
			weight_id += 1
			w1 = utils.get_variable(weights[weight_id], w1_name)
			weight_id += 1
			w2 = utils.get_variable(weights[weight_id], w2_name)
			weight_id += 1	
			b1 = utils.get_variable(weights[weight_id], b1_name)
			weight_id += 1
			b2 = utils.get_variable(weights[weight_id], b2_name)
			weight_id += 1				
		else:
			wd = utils.weight_variable([2, 2, features//2, features], stddev, wd_name)
			bd = utils.bias_variable([features//2], bd_name)
			w1 = utils.weight_variable([filter_size, filter_size, features, features//2], stddev, w1_name)
			w2 = utils.weight_variable([filter_size, filter_size, features//2, features//2], stddev, w2_name)
			b1 = utils.bias_variable([features//2], b1_name)
			b2 = utils.bias_variable([features//2], b2_name)		
		h_deconv = tf.nn.relu(utils.conv2d_transpose_strided(in_node, wd, bd, keep_prob = keep_prob))
		h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer], h_deconv)
		conv1 = utils.conv2d_basic(h_deconv_concat, w1, b1, keep_prob)
		h_conv = tf.nn.relu(conv1)
		conv2 = utils.conv2d_basic(h_conv, w2, b2, keep_prob)

		in_node = tf.nn.relu(conv2, relu_name)
		if debug: 
			utils.add_activation_summary(in_node)
			utils.add_to_image_summary(utils.get_image_summary(in_node,  relu_name + '_image'))


		new_variables.extend((wd, bd, w1, w2, b1, b2))
		layer_id+=1
	return in_node, new_variables, layer_id, weight_id

def unet(image, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore = False, weights = None):
	channels = image.get_shape().as_list()[-1]
	dw_h_convs = {}
	variables = []
	pools = {}
	in_node = image

	# downsample layer
	layer_id = 0
	weight_id = 0
	for layer in range(0, num_of_layers):
		features = 2**layer*num_of_feature
		stddev = np.sqrt(float(2) / (filter_size**2 * features))

		w1_name = name + '_layer_' + str(layer_id) + '_w_0'
		w2_name = name + '_layer_' + str(layer_id) + '_w_1'
		b1_name = name + '_layer_' + str(layer_id) + '_b_0'
		b2_name = name + '_layer_' + str(layer_id) + '_b_1'
		relu_name = name + '_layer_' + str(layer_id) + '_feat'
		if layer == 0:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:
			    w1 = utils.weight_variable([filter_size, filter_size, channels, features], stddev, w1_name)
		else:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:			
				w1 = utils.weight_variable([filter_size, filter_size, features//2, features], stddev, w1_name)

		if restore == True:
			w2 = utils.get_variable(weights[weight_id], w2_name)
			weight_id+=1
			b1 = utils.get_variable(weights[weight_id], b1_name)
			weight_id+=1
			b2 = utils.get_variable(weights[weight_id], b2_name)
			weight_id+=1
		else:
			w2 = utils.weight_variable([filter_size, filter_size, features, features], stddev, w2_name)
			b1 = utils.bias_variable([features], b1_name)
			b2 = utils.bias_variable([features], b2_name)

		conv1 = utils.conv2d_basic(in_node, w1, b1, keep_prob)
		tmp_h_conv = tf.nn.relu(conv1)
		conv2 = utils.conv2d_basic(tmp_h_conv, w2, b2, keep_prob)
		
		dw_h_convs[layer] = tf.nn.relu(conv2, relu_name)      

		if layer < num_of_layers-1:
		    pools[layer] = utils.max_pool_2x2(dw_h_convs[layer])
		    in_node = pools[layer]
		 
		
		if debug: 
			utils.add_activation_summary(dw_h_convs[layer])
			utils.add_to_image_summary(utils.get_image_summary(dw_h_convs[layer],  relu_name + '_image'))


		variables.extend((w1, w2, b1, b2))

		layer_id+=1

	in_node = dw_h_convs[num_of_layers - 1]

     # upsample layer
	for layer in range(num_of_layers - 2, -1, -1):
		features = 2 ** (layer + 1) * num_of_feature
		stddev = np.sqrt(float(2) / (filter_size**2 * features))

		wd_name = name + '_layer_up' + str(layer_id) + '_w'
		bd_name = name + '_layer_up' + str(layer_id) + '_b'
		w1_name = name + '_layer_up_conv' + str(layer_id) + '_w0'
		w2_name = name + '_layer_up_conv' + str(layer_id) + '_w1'
		b1_name = name + '_layer_up_conv' + str(layer_id) + '_b0'
		b2_name = name + '_layer_up_conv' + str(layer_id) + '_b1'
		relu_name = name + '_layer_up_conv' + str(layer_id) + '_feat'

		# pooling size is 2
		if restore == True:
			wd = utils.get_variable(weights[weight_id], wd_name)
			weight_id += 1
			bd = utils.get_variable(weights[weight_id], bd_name)
			weight_id += 1
			w1 = utils.get_variable(weights[weight_id], w1_name)
			weight_id += 1
			w2 = utils.get_variable(weights[weight_id], w2_name)
			weight_id += 1	
			b1 = utils.get_variable(weights[weight_id], b1_name)
			weight_id += 1
			b2 = utils.get_variable(weights[weight_id], b2_name)
			weight_id += 1				
		else:
			wd = utils.weight_variable([2, 2, features//2, features], stddev, wd_name)
			bd = utils.bias_variable([features//2], bd_name)
			w1 = utils.weight_variable([filter_size, filter_size, features, features//2], stddev, w1_name)
			w2 = utils.weight_variable([filter_size, filter_size, features//2, features//2], stddev, w2_name)
			b1 = utils.bias_variable([features//2], b1_name)
			b2 = utils.bias_variable([features//2], b2_name)	
					
		h_deconv = tf.nn.relu(utils.conv2d_transpose_strided(in_node, wd, bd))

		# h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer], h_deconv, tf.shape(image)[0])
		h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer], h_deconv)

		conv1 = utils.conv2d_basic(h_deconv_concat, w1, b1, keep_prob)
		h_conv = tf.nn.relu(conv1)
		conv2 = utils.conv2d_basic(h_conv, w2, b2, keep_prob)

		in_node = tf.nn.relu(conv2, relu_name)
		if debug: 

			utils.add_to_image_summary(utils.get_image_summary(in_node,  relu_name + '_image'))
			utils.add_to_image_summary(utils.get_image_summary(conv2,  relu_name + '_image'))


		variables.extend((wd, bd, w1, w2, b1, b2))
		layer_id+=1
    
	w_name = name + '_final_layer_' + str(layer_id) + '_w'
	b_name = name + '_final_layer_' + str(layer_id) + '_b'
	relu_name =  name + '_final_layer_' + str(layer_id) + '_feat'
	if restore == True:
		w = utils.get_variable(weights[weight_id], w_name)
		weight_id += 1
		b = utils.get_variable(weights[weight_id], b_name)
		weight_id += 1		
	else:
		w = utils.weight_variable([1, 1, num_of_feature, n_class], stddev, w_name)
		b = utils.bias_variable([n_class], b_name)

	y_conv = tf.nn.relu(utils.conv2d_basic(in_node, w, b), relu_name)
	variables.extend((w, b))
	if debug: 
		utils.add_activation_summary(y_conv)
		utils.add_to_image_summary(utils.get_image_summary(y_conv,  relu_name + '_image'))


	return y_conv, variables, layer_id, dw_h_convs

def autoencorder_unet(image, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore = False, weights = None):
	channels = image.get_shape().as_list()[-1]
	y_conv, variables, layer_id = unet(image, channels, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore, weights)
	return y_conv, variables

def AutoencorderCLustering(image, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, Class, restore = False, weights = None):
	channels = image.get_shape().as_list()[-1]
	dw_h_convs = {}
	variables = []
	pools = {}
	in_node = image

	# downsample layer
	layer_id = 0
	weight_id = 0
	for layer in range(0, num_of_layers):
		features = 2**layer*num_of_feature
		stddev = np.sqrt(float(2) / (filter_size**2 * features))

		w1_name = name + '_layer_' + str(layer_id) + '_w_0'
		w2_name = name + '_layer_' + str(layer_id) + '_w_1'
		b1_name = name + '_layer_' + str(layer_id) + '_b_0'
		b2_name = name + '_layer_' + str(layer_id) + '_b_1'
		relu_name = name + '_layer_' + str(layer_id) + '_feat'
		if layer == 0:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:
			    w1 = utils.weight_variable([filter_size, filter_size, channels, features], stddev, w1_name)
		else:
			if restore == True:
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id+=1
			else:			
				w1 = utils.weight_variable([filter_size, filter_size, features//2, features], stddev, w1_name)

		if restore == True:
			w2 = utils.get_variable(weights[weight_id], w2_name)
			weight_id+=1
			b1 = utils.get_variable(weights[weight_id], b1_name)
			weight_id+=1
			b2 = utils.get_variable(weights[weight_id], b2_name)
			weight_id+=1
		else:
			w2 = utils.weight_variable([filter_size, filter_size, features, features], stddev, w2_name)
			b1 = utils.bias_variable([features], b1_name)
			b2 = utils.bias_variable([features], b2_name)

		conv1 = utils.conv2d_basic(in_node, w1, b1, keep_prob)
		tmp_h_conv = tf.nn.relu(conv1)
		conv2 = utils.conv2d_basic(tmp_h_conv, w2, b2, keep_prob)
		
		dw_h_convs[layer] = tf.nn.relu(conv2, relu_name)      

		if layer < num_of_layers-1:
		    pools[layer] = utils.max_pool_2x2(dw_h_convs[layer])
		    in_node = pools[layer]
		 
		
		if debug: 
			utils.add_activation_summary(dw_h_convs[layer])
			utils.add_to_image_summary(utils.get_image_summary(dw_h_convs[layer],  relu_name + '_image'))


		variables.extend((w1, w2, b1, b2))

		layer_id+=1
	EncodedNode = dw_h_convs[num_of_layers - 1]

    # upsample layer
	Representation = []
	for k in range(Class):
		in_node = EncodedNode
		for layer in range(num_of_layers - 2, -1, -1):
			features = 2 ** (layer + 1) * num_of_feature
			stddev = np.sqrt(float(2) / (filter_size**2 * features))

			wd_name = name + '_layer_up' + str(layer_id) + '_w' + 'Class' + str(k)
			bd_name = name + '_layer_up' + str(layer_id) + '_b' + 'Class' + str(k)
			w1_name = name + '_layer_up_conv' + str(layer_id) + '_w0' + 'Class' + str(k)
			w2_name = name + '_layer_up_conv' + str(layer_id) + '_w1' + 'Class' + str(k)
			b1_name = name + '_layer_up_conv' + str(layer_id) + '_b0' + 'Class' + str(k)
			b2_name = name + '_layer_up_conv' + str(layer_id) + '_b1' + 'Class' + str(k)
			relu_name = name + '_layer_up_conv' + str(layer_id) + '_feat' + 'Class' + str(k)

			# pooling size is 2
			if restore == True:
				wd = utils.get_variable(weights[weight_id], wd_name)
				weight_id += 1
				bd = utils.get_variable(weights[weight_id], bd_name)
				weight_id += 1
				w1 = utils.get_variable(weights[weight_id], w1_name)
				weight_id += 1
				w2 = utils.get_variable(weights[weight_id], w2_name)
				weight_id += 1	
				b1 = utils.get_variable(weights[weight_id], b1_name)
				weight_id += 1
				b2 = utils.get_variable(weights[weight_id], b2_name)
				weight_id += 1				
			else:
				wd = utils.weight_variable([2, 2, features//2, features], stddev, wd_name)
				bd = utils.bias_variable([features//2], bd_name)
				w1 = utils.weight_variable([filter_size, filter_size, features, features//2], stddev, w1_name)
				w2 = utils.weight_variable([filter_size, filter_size, features//2, features//2], stddev, w2_name)
				b1 = utils.bias_variable([features//2], b1_name)
				b2 = utils.bias_variable([features//2], b2_name)	
						
			h_deconv = tf.nn.relu(utils.conv2d_transpose_strided(in_node, wd, bd))

			# h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer], h_deconv, tf.shape(image)[0])
			h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer], h_deconv)

			conv1 = utils.conv2d_basic(h_deconv_concat, w1, b1, keep_prob)
			h_conv = tf.nn.relu(conv1)
			conv2 = utils.conv2d_basic(h_conv, w2, b2, keep_prob)

			in_node = tf.nn.relu(conv2, relu_name)
			if debug: 

				utils.add_to_image_summary(utils.get_image_summary(in_node,  relu_name + '_image'))
				utils.add_to_image_summary(utils.get_image_summary(conv2,  relu_name + '_image'))


			variables.extend((wd, bd, w1, w2, b1, b2))
			layer_id+=1
	    
		w_name = name + '_final_layer_' + str(layer_id) + '_w' + str(k)
		b_name = name + '_final_layer_' + str(layer_id) + '_b' + str(k)
		relu_name =  name + '_final_layer_' + str(layer_id) + '_feat' + str(k)

		if restore == True:
			w = utils.get_variable(weights[weight_id], w_name)
			weight_id += 1
			b = utils.get_variable(weights[weight_id], b_name)
			weight_id += 1		
		else:
			w = utils.weight_variable([1, 1, num_of_feature, channels], stddev, w_name)
			b = utils.bias_variable([channels], b_name)

		y_conv = tf.nn.relu(utils.conv2d_basic(in_node, w, b), relu_name)		

		variables.extend((w, b))
		if debug: 
			utils.add_activation_summary(y_conv)
			utils.add_to_image_summary(utils.get_image_summary(y_conv, relu_name))

		Representation.append(y_conv)

	return Representation, variables, dw_h_convs

def _trans_layer(inter_feat, channel, shape):
	num_of_feature = inter_feat.get_shape().as_list()[-1]
	intersec = num_of_feature / (channel)
	feat = []
	for i in range(channel):
		feat.append(tf.reduce_mean(inter_feat[:, :, :, i * intersec : i * intersec + intersec], 3))
	tran_y_feat = tf.image.resize_images(tf.stack(feat, 3), shape)
	return tran_y_feat

def autoencorder_antn(image, n_class, filter_size, num_of_feature, num_of_layers, 
					  keep_prob, debug, restore = False, shared_weights = None, M_weights = None, AE_weights = None):
	stddev = 0.02
	channels = image.get_shape().as_list()[-1]
	with tf.name_scope("shared-network"):
		name = 'shared'
		inter_feat, shared_variables, layer_id, weight_id = unet_downsample(image, filter_size, num_of_feature, num_of_layers,
		 												   			 keep_prob, name, debug, restore, shared_weights)	

	with tf.name_scope("main-network"):
		name = 'main-network'
		M_feat, M_variables, M_layer_id, M_weight_id = unet_upsample(image, inter_feat, shared_variables, layer_id, weight_id, filter_size, 
																   		 num_of_feature, num_of_layers, keep_prob, name, debug, 
																   		 restore, M_weights)
		w_name = name + '_final_layer_' + str(M_layer_id) + '_w'
		b_name = name + '_final_layer_' + str(M_layer_id) + '_b'
		relu_name =  name + '_final_layer_' + str(M_layer_id) + '_feat'
		if restore == True:
			w = utils.get_variable(M_weights[M_weight_id], w_name)
			M_weight_id += 1
			b = utils.get_variable(M_weights[M_weight_id], b_name)
			M_weight_id += 1		
		else:
			w = utils.weight_variable([1, 1, num_of_feature, n_class], stddev, w_name)
			M_weight_id+=1
			b = utils.bias_variable([n_class], b_name)
			M_weight_id+=1
		y_conv = utils.conv2d_basic(M_feat, w, b, keep_prob)
		y_conv_relu = tf.nn.relu(y_conv)
		clean_y_out = tf.reshape(tf.nn.softmax(tf.reshape(y_conv, [-1, n_class])), tf.shape(y_conv), 'segmentation_map')
		M_variables.extend((w, b))
		if debug:
			utils.add_activation_summary(clean_y_out)
			utils.add_to_image_summary(clean_y_out)		
		M_layer_id+= 1

	with tf.name_scope("auto-encoder"):
		name = 'auto-encoder'
		# AE_conv, AE_variables, AE_layer_id, AE_weight_id = unet_upsample(image, inter_feat, shared_variables, layer_id, weight_id, filter_size, 
		# 														   num_of_feature, num_of_layers, keep_prob, name, debug, 
		# 														   restore, weights)

		w_name = name + '_final_layer_' + str(M_layer_id) + '_w'
		b_name = name + '_final_layer_' + str(M_layer_id) + '_b'
		relu_name =  name + '_final_layer_' + str(M_layer_id) + '_feat'
		# contrating layer of main network as input 
		# if restore == True:
		# 	w = utils.get_variable(weights[AE_weight_id], w_name)
		# 	AE_weight_id += 1
		# 	b = utils.get_variable(weights[AE_weight_id], b_name)
		# 	AE_weight_id += 1		
		# else:
		# w = utils.weight_variable([1, 1, num_of_feature, channels], stddev, w_name)
		# AE_weight_id+=1
		# b = utils.bias_variable([channels], b_name)
		# AE_weight_id+=1
		# AE_feat = tf.nn.relu(utils.conv2d_basic(AE_conv, w, b, keep_prob), relu_name)
		# AE_variables.extend((w, b))
		# AE_layer_id+=1
		# last layer of main network as input 
		AE_variables = []
		w = utils.weight_variable([1, 1, num_of_feature, channels], stddev, w_name)
		b = utils.bias_variable([channels], b_name)
		AE_feat = tf.nn.relu(utils.conv2d_basic(M_feat, w, b, keep_prob), relu_name)
		AE_variables.extend((w, b))
		if debug: 
			utils.add_activation_summary(AE_feat)
			utils.add_to_image_summary(utils.get_image_summary(AE_feat,  relu_name + '_image'))	

	with tf.name_scope("trans-layer"):
		# trans_variables = []
		name = 'trans_layer'
		# wd_name = name  + str(layer_id) + '_w'
		# bd_name = name  + str(layer_id) + '_b'
		# features = 2 ** (num_of_layers - 1) * num_of_feature
		# wd = utils.weight_variable([2, 2, n_class * n_class, features], stddev, wd_name)
		# bd = utils.bias_variable([features//2], bd_name)
		# output_shape = [tf.shape(inter_feat)[0], tf.shape(inter_feat)[1] * 4, tf.shape(inter_feat)[2] * 4, n_class * n_class]
		# tran_y_feat = tf.nn.relu(utils.conv2d_transpose_strided(in_node, wd, bd, output_shape=output_shape, keep_prob = keep_prob))
		# trans_variables.extend((wd, bd))

		tran_y_feat = _trans_layer(inter_feat[0], n_class * n_class, [image.get_shape().as_list()[1], image.get_shape().as_list()[2]])
		class_tran_y_out = []
		for i in range (n_class):
			class_tran_y_out.append(tf.reshape(tf.nn.softmax(tf.reshape(tran_y_feat[:, :, :, i * n_class : (i * n_class + n_class)], [-1, n_class])),
									tf.shape(clean_y_out), name = 'tran_map' + str(i)))

		tran_map = tf.concat(class_tran_y_out, 3, name = 'tran_map')
		if debug:
			for i in range(n_class):
				for j in range(n_class):
					# utils.add_activation_summary(utils.get_image_summary(tran_map,  str(i) + '_to_'+ str(j), i * n_class + j))
					utils.add_activation_summary(tran_map[:, :, :, i * n_class + j])
					utils.add_to_image_summary(utils.get_image_summary(tran_map,  str(i) + '_to_'+ str(j), i * n_class + j))



	with tf.name_scope("noisy-map-layer"):
		noise_y_out = tf.reshape(tf.matmul(tf.reshape(clean_y_out, [-1, 1, n_class]), tf.reshape(tran_map, [-1, n_class, n_class])), 
								tf.shape(clean_y_out), name = 'noise_output')

		# summary
		if debug: 
			utils.add_activation_summary(noise_y_out)
			utils.add_to_image_summary(noise_y_out)

	return noise_y_out, clean_y_out, y_conv, tran_y_feat, tran_map, AE_feat, shared_variables, AE_variables, M_variables, inter_feat

def NTN(image, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore = False, weights = None, Unsupervised = False):

	with tf.name_scope("u-net"):
		
		y_conv, variables, layer_id, dw_h_convs = unet(image, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore, weights)
		clean_y_out = tf.reshape(tf.nn.softmax(tf.reshape(y_conv, [-1, n_class])), tf.shape(y_conv), 'segmentation_map') #softmax y output
		
		# summary
		if debug: 
			utils.add_activation_summary(clean_y_out)
			utils.add_to_image_summary(clean_y_out)


	with tf.name_scope("trans-layer"):
		if Unsupervised == False:
			w = utils.get_variable(np.reshape(np.eye(n_class), [1, 1, n_class, n_class]), 'trans-prob-weight')
		else:
			w = utils.weight_constant(np.reshape(np.ones((n_class, n_class)), [1, 1, n_class, n_class]) * 1. /  n_class, 'trans-prob-weight')
		TransProbVar = []
		noise_y_out = tf.nn.conv2d(clean_y_out, w, strides=[1, 1, 1, 1], padding="SAME", name = "NoisySegMap")
		TransProbVar.append(w)
		if debug: 
			utils.add_activation_summary(noise_y_out)
			utils.add_to_image_summary(noise_y_out)


	with tf.name_scope("MapTransProb"):
		if Unsupervised == False:
			MIN = tf.zeros([n_class, n_class], dtype=tf.float32)
			MAX = tf.ones([n_class, n_class], dtype=tf.float32)
			I_MIN = tf.maximum(w[0, 0], MIN, name = "MAXIMUM")
			I_MAX = tf.minimum(I_MIN, MAX, name = "MINIMUM")
			B = tf.reshape(I_MAX / tf.reshape((tf.reduce_sum(I_MAX, 1)), [n_class, 1]), [1, 1, n_class, n_class])
			MapTransProb = tf.assign(w, B)
		else:
			MapTransProb = None

	return noise_y_out, clean_y_out, MapTransProb, variables, TransProbVar, dw_h_convs 

def l2_loss(s, t):
	""" L2 loss function. """
	d = s - t
	x = d * d
	loss = tf.reduce_sum(x, 1)
	return loss

def CreateAutoencoderClustering(image, Label, Class, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, reg_weight = None, restore = False, weights = None):

	with tf.name_scope("Autoencoder"):
		Representation, variables, dw_h_convs = AutoencorderCLustering(image, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, Class, restore, weights)


	with tf.name_scope("loss"):
		UnregLoss = []
		AllLoss = 0
		for k, y_conv in enumerate(Representation): 
			UnregLossClass = tf.reduce_mean(l2_loss(tf.reshape(Label[k], [-1, channel]), tf.reshape(y_conv, [-1, channel])), name = 'UnregLossClass' + str(k)) 
			UnregLoss.append(UnregLossClass)
			AllLoss+= UnregLossClass

		weight_decay = 0
		if reg_weight != None:
			for var in variables:
				weight_decay = weight_decay + tf.nn.l2_loss(var)

		AllLoss = tf.add(AllLoss, reg_weight * weight_decay, name = 'loss')

		if debug: 
			utils.add_scalar_summary(AllLoss)
			for loss in UnregLoss:
				utils.add_scalar_summary(loss)

	return Representation, variables, dw_h_convs, UnregLoss, AllLoss



def create_autoencoder_antn(image, decoded_image, label, posterior_prob, n_class, 
							filter_size, num_of_feature, num_of_layers, 
							keep_prob, debug, constrain_weight = 0, reg_weight = 0, restore = False, 
							shared_weights = None, M_weights = None, AE_weights = None):

	channel = image.get_shape().as_list()[-1]
	with tf.name_scope("autoencoder-antn"):
		noise_y_out, clean_y_out, y_conv, tran_y_feat, tran_map, AE_conv, shared_variables, AE_variables, M_variables, inter_feat = autoencorder_antn(image, n_class, 
    																												 			  		  filter_size, num_of_feature, 
    																												 			  		  num_of_layers, keep_prob, 
    																												 			  		  debug, restore, shared_weights,
    																												 			  		  M_weights, AE_weights)


	with tf.name_scope("loss"):
		# M_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(tf.argmax(label, 3), [-1]), 
		# 																	   logits = tf.reshape(noise_y_out, [-1, n_class])),
		# 																	   name = 'M_loss') 
		M_loss = tf.reduce_mean(utils.cross_entropy(tf.cast(tf.reshape(label, [-1, n_class]), tf.float32), tf.reshape(noise_y_out, [-1, n_class])), name = 'M_loss') 

		M_hidden_loss = tf.reduce_mean(l2_loss(tf.cast(tf.reshape(posterior_prob, [-1, n_class]), tf.float32), tf.reshape(clean_y_out, [-1, n_class]))) 

		# M_hidden_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(tf.argmax(posterior_prob, 3), [-1]), 
		# 																	   logits = tf.reshape(y_conv, [-1, n_class])),
		# 																	   name = 'M_hidden_loss') 
		M_hidden_loss_reg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(tf.argmax(label, 3), [-1]), 
																											  logits = tf.reshape(y_conv, [-1, n_class])),
																											  name = 'M_hidden_loss_reg') 

		AE_loss = tf.reduce_mean(l2_loss(tf.reshape(decoded_image, [-1, channel]), tf.reshape(AE_conv, [-1, channel]))) 
		
		weight_decay1 = 0
		weight_decay2 = 0
		weight_decay3 = 0
		if reg_weight != None:
			for var in (shared_variables + M_variables + AE_variables):
				weight_decay1 = weight_decay1 + tf.nn.l2_loss(var)

		if reg_weight != None:
			for var in (shared_variables  + AE_variables):
				weight_decay2 = weight_decay2 + tf.nn.l2_loss(var)

		if reg_weight != None:
			for var in (shared_variables + M_variables):
				weight_decay3 = weight_decay3 + tf.nn.l2_loss(var)								
		
		loss = tf.reduce_sum((1 - constrain_weight) * M_hidden_loss + constrain_weight* M_hidden_loss_reg + AE_loss  + reg_weight*weight_decay1, name = 'loss')
		M_hidden_loss = tf.reduce_sum((1 - constrain_weight)* M_hidden_loss + constrain_weight * M_hidden_loss_reg + reg_weight*weight_decay3, name = 'M_hidden_loss')
		AE_loss = tf.reduce_sum(AE_loss + reg_weight*weight_decay2, name = 'AE_loss')

		# M_loss = tf.reduce_sum(M_loss + reg_weight*weight_decay, name = 'M_loss')
		# AE_loss = tf.reduce_sum(AE_loss + reg_weight*weight_decay, name = 'AE_loss')

		if debug: 
			utils.add_scalar_summary(loss)
			utils.add_scalar_summary(M_loss)
			utils.add_scalar_summary(AE_loss)
			for var in (shared_variables + AE_variables + M_variables):
				utils.add_to_regularization_and_summary(var)
	return noise_y_out, clean_y_out, tran_y_feat, tran_map, AE_conv, loss, M_loss, M_hidden_loss, AE_loss, shared_variables, AE_variables, M_variables, inter_feat


def create_unet(image, label, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, reg_weight = 0.001, debug = True,  restore = False, weights = None, ClassWeights = [1, 1, 1]):

	with tf.name_scope("u-net"):
		y_conv, variables, layer_id, dw_h_convs = unet(image, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore, weights)
		clean_y_out = tf.reshape(tf.nn.softmax(tf.reshape(y_conv, [-1, n_class])), tf.shape(y_conv), 'segmentation_map') #softmax y output
		
		# summary
		if debug: 
			utils.add_activation_summary(clean_y_out)
			utils.add_to_image_summary(clean_y_out)
			for var in variables:
				utils.add_to_regularization_and_summary(var)

	with tf.name_scope("loss"):
		# adding class weight
		# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(label, [-1]), logits = tf.multiply(tf.reshape(y_conv, [-1, n_class]), ClassWeights))) 

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(label, [-1, n_class]), 
																	  logits = tf.reshape(y_conv, [-1, n_class]))) 
		weight_decay = 0
		if reg_weight != None:
			for var in variables:
				weight_decay = weight_decay + tf.nn.l2_loss(var)
		
		loss = tf.reduce_sum(loss + reg_weight*weight_decay, name = 'loss')		
		if debug: 
			utils.add_scalar_summary(loss)

	return loss, clean_y_out, variables, dw_h_convs



def create_ntn(image, label, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, name, reg_weight, debug, restore = False, weights = None, Unsupervised = False, ClassWeights = [1, 0.2, 1]):
	with tf.name_scope("NTN"):
		noise_y_out, clean_y_out, MapTransProb, variables, TransProbVar, dw_h_convs = NTN(image, n_class, filter_size, num_of_feature, num_of_layers, 
																			  keep_prob, name, debug, restore, weights, Unsupervised)
	
	# summary
	if debug: 		
		for var in variables:
			utils.add_to_regularization_and_summary(var)
		for var in TransProbVar:
			utils.add_to_regularization_and_summary(var)

	with tf.name_scope("loss"):
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(label, [-1, n_class]), 
		# 															  logits = tf.multiply(tf.reshape(y_conv, [-1, n_class]), ClassWeights))) 
		loss = utils.cross_entropy(tf.cast(tf.reshape(label, [-1, n_class]), tf.float32), tf.reshape(noise_y_out, [-1, n_class]))
		weight_decay = 0
		if reg_weight != None:
			for var in TransProbVar:
				weight_decay = weight_decay + tf.nn.l2_loss(var)
			for var in variables:
				weight_decay = weight_decay + tf.nn.l2_loss(var)
		
		loss = tf.reduce_sum(loss + reg_weight*weight_decay, name = 'loss')		
		if debug: 
			utils.add_scalar_summary(loss)

	return loss, noise_y_out, clean_y_out, MapTransProb, variables, TransProbVar, dw_h_convs


def create_autoencoder(image, label, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, reg_weight = None, restore = False, weights = None):
	channel = image.get_shape().as_list()[-1]
	with tf.name_scope("autoencoder-u-net"):
		y_conv, variables = autoencorder_unet(image, filter_size, num_of_feature, num_of_layers, keep_prob, name, debug, restore, weights)
		# summary
		if debug: 
			for var in variables:
				utils.add_to_regularization_and_summary(var)

	with tf.name_scope("loss"):
		unreg_loss = tf.reduce_mean(l2_loss(tf.reshape(label, [-1, channel]), tf.reshape(y_conv, [-1, channel]))) 
		weight_decay = 0
		if reg_weight != None:
			for var in variables:
				weight_decay = weight_decay + tf.nn.l2_loss(var)
		loss = tf.add(unreg_loss, reg_weight * weight_decay, name = 'loss')
		if debug: 
			utils.add_scalar_summary(loss)

	return loss, y_conv, variables

# noisy_label: [batch, row, col, class]
def posterior_prob_hidden(noisy_label, clean_map, trans_map):
	n_class = clean_map.shape[-1]
	mask = np.tile(noisy_label, [1, 1, 1, n_class])
	obs_trans_map = np.reshape(trans_map[np.where(mask == 1)], [-1, n_class])
	flat_clean_map = np.reshape(clean_map, [-1, n_class])
	clean_network_hidden = np.divide(np.multiply(obs_trans_map, flat_clean_map), 
		                             np.clip(np.reshape(np.sum(np.multiply(obs_trans_map,
		                             				  		   flat_clean_map), 1), [-1, 1]), 
		                                              1e-6, 1.0))
	
	trans_network_hidden = np.zeros(np.shape(mask))
	trans_network_hidden[np.where(mask == 1)] = np.reshape(clean_network_hidden, [-1])
	rest_prob = (1 - np.reshape(clean_network_hidden, [-1, 1])) / (n_class - 1)
	trans_network_hidden[np.where(mask == 0)] = np.reshape(np.tile(rest_prob, [1, n_class - 1]), [-1])
	trans_network_hidden = np.reshape(trans_network_hidden, [-1, n_class * n_class])

	clean_network_hidden = np.reshape(clean_network_hidden, [clean_map.shape[0], clean_map.shape[1], clean_map.shape[2], clean_map.shape[3]])
	trans_network_hidden = np.reshape(trans_network_hidden, [trans_map.shape[0], trans_map.shape[1], trans_map.shape[2], trans_map.shape[3]])
	return clean_network_hidden, trans_network_hidden

def normalize(image):
	batch = image.shape[0]
	row = image.shape[1]
	col = image.shape[2]
	channel = image.shape[3]
	I_min = np.reshape(np.amin(np.reshape(image, [-1, row * col * channel]), axis = 1), [batch, 1])
	I_max = np.reshape(np.amax(np.reshape(image, [-1, row * col * channel]), axis = 1), [batch, 1])
	norm_I = np.reshape((np.reshape(image, [-1, row * col * channel]) - I_min) / np.float32(I_max - I_min), [-1, row, col, channel])
	return norm_I

# image: [batch row col channel]
# clean_network_hidden: 
# trans_network_hidden
def ANTN(image, label, n_class, filter_size, num_of_branch, 
		 num_of_feature, num_of_layers, clean_network_hidden, 
		 trans_network_hidden, debug, keep_prob = 1.0, 
		 restore_clean = False, restore_tran = False, weights = None, 
		 all_tran_var = None):
	with tf.name_scope("clean-network"):
		clean_y_feat, clean_var, layer_id = unet(image, n_class, filter_size, num_of_feature, num_of_layers, keep_prob, 'main', debug, restore_clean, weights)
		clean_y_out = tf.clip_by_value(tf.reshape(tf.nn.softmax(tf.reshape(clean_y_feat, [-1, n_class])), tf.shape(clean_y_feat), 'clean_map'),
									   1e-6, 1.0) #softmax y output
		# summary
		if debug: 
			utils.add_activation_summary(clean_y_out)
			utils.add_to_image_summary(clean_y_out)
			for var in clean_var:
				utils.add_to_regularization_and_summary(var)

	# branch process
	all_tran_y_out = []
	all_tran_var = []
	for branch in range(num_of_branch):
		with tf.name_scope("transition-network" + str(branch)):
			if restore_trans == True:
				tran_y_feat, tran_var, layer_id = unet(image, n_class * n_class, filter_size, num_of_feature, num_of_layers, keep_prob, 'tran' + str(branch), debug, restore_tran, all_tran_var[branch])
			else:
				tran_y_feat, tran_var, layer_id = unet(image, n_class * n_class, filter_size, num_of_feature, num_of_layers, keep_prob, 'tran' + str(branch), debug)
			class_tran_y_out = []
			for i in range (n_class):
				class_tran_y_out.append(tf.reshape(tf.nn.softmax(tf.reshape(tran_y_feat[:, :, :, i * n_class : (i * n_class + n_class)], 
		        								  	    [-1, n_class])), tf.shape(clean_y_feat), name = 'tran_map' + str(i)))


			tran_y_out = tf.clip_by_value(tf.concat(class_tran_y_out, 3, name = 'tran_map'),
										  1e-6, 1.0)	
			all_tran_y_out.append(tran_y_out)
			all_tran_var.append(tran_var)
			# summary
			if debug:
				# for clss in class_tran_y_out:
				# 	utils.add_activation_summary(clss)
				# 	utils.add_to_image_summary(clss)
				# for var in tran_var:
				# 	utils.add_to_regularization_and_summary(var)
				for i in range(n_class):
					for j in range(n_class):
						z = tf.identity(tran_y_out[:, :, :, i], name = str(i) + 'to' + str(j))
						utils.add_activation_summary(z)
				for var in tran_var:
					utils.add_to_regularization_and_summary(var)
	# branch process
	all_noise_y_out = []
	for branch in range(num_of_branch):
		with tf.name_scope("integration" + str(branch)):	
			noise_y_out = tf.reshape(tf.matmul(tf.reshape(clean_y_out, [-1, 1, n_class]), tf.reshape(all_tran_y_out[branch], [-1, n_class, n_class])), 
									tf.shape(clean_y_feat), name = 'noise_output')
			all_noise_y_out.append(noise_y_out)
			# summary
			if debug: 
				utils.add_activation_summary(noise_y_out)
				utils.add_to_image_summary(noise_y_out)

	with tf.name_scope("loss"):
		# clean_net_loss = utils.cross_entropy(tf.reshape(clean_network_hidden, [-1, n_class]), tf.reshape(clean_y_out, [-1, n_class]), 'clean_net_loss')
		# noise_loss = utils.cross_entropy(tf.cast(tf.reshape(label, [-1, n_class]), tf.float32), tf.reshape(noise_y_out, [-1, n_class]), 'noise_loss')
		# trans_net_loss = utils.cross_entropy(tf.reshape(trans_network_hidden, [-1, n_class * n_class]), tf.reshape(tran_y_out, [-1, n_class * n_class]), 'trans_net_loss')

		# clean_net_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(clean_network_hidden, [-1, n_class]), 
		# 																		logits = tf.reshape(clean_y_feat, [-1, n_class])), 
		# 																		name = 'clean_net_loss')

		clean_net_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(clean_network_hidden, [-1]), 
																					   logits = tf.reshape(clean_y_feat, [-1, n_class])), 
																					   name = 'clean_net_loss')
		# branch process
		all_noise_loss = []
		all_trans_net_loss = []
		for branch in range(num_of_branch):
			noise_loss = utils.cross_entropy(tf.cast(tf.reshape(label[branch], [-1, n_class]), tf.float32), 
											 tf.reshape(all_noise_y_out[branch], [-1, n_class]), 
											 'noise_loss')
			trans_net_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(trans_network_hidden[branch], [-1, n_class * n_class]), 
																					logits = tf.reshape(all_tran_y_out[branch], [-1,  n_class * n_class])), 
																					name = 'trans_net_loss')
			all_noise_loss.append(noise_loss)
			all_trans_net_loss.append(trans_net_loss)

			if debug: 
				utils.add_scalar_summary(noise_loss)
	return all_noise_loss, clean_net_loss, all_trans_net_loss, clean_y_out, all_tran_y_out, all_noise_y_out, clean_var, all_tran_var



