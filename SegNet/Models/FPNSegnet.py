from keras.models import *
from keras.layers import *

import os
file_path = os.path.dirname( os.path.abspath(__file__) )
FPN_Weights_path = file_path+"/data/vgg16_weights_th_dim_ordering_th_kernels.h5"

def bottleneck_layer(x_in, num_filters, stride=1):

  x = Conv2D(num_filters, (1,1))(x_in)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(num_filters, (3,3), strides=(stride,stride))
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(4*num_filters, (1,1))
  x = BatchNormalization()(x)

  x = concatenate([x, x_in])
  x = Activation('relu')(x)

  return x

def bottleneck_aggregate(x_in, num_filters, num_blocks, stride):
  strides = [stride] + [1]*(num_blocks-1)
  
  x_out = []
  for stride in strides:
    x = bottleneck_layer(x_in, num_filters, stride)
    x_out.append(x)
  
  return x_out

def upsample_add(x, y):
  height = int(y.shape[0])
  width = int(y.shape[1])

  z = UpSampling2D(size=(height, width))(x)
  z = Add([z, y])

  return z

def FPNSegnet(n_classes, input_height=416, input_width=608, fpn_level=3):

  # FPN
  img_input = Input(shape=(3,input_height,input_width))

  x = Conv2D( 64, (7,7), strides=(2,2), padding="same" )(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)

  num_blocks = [2,2,2,2]
  c2 = bottleneck_aggregate(x, 64, num_blocks[0], 1)
  c3 = bottleneck_aggregate(c2, 128, num_blocks[1], 2)
  c4 = bottleneck_aggregate(c3, 256, num_blocks[2], 2)
  c5 = bottleneck_aggregate(c4, 512, num_blocks[3], 2)

  c4_lat = Conv2D(256, (1,1))(c4)
  c3_lat = Conv2D(256, (1,1))(c3)
  c2_lat = Conv2D(256, (1,1))(c2)

  p5 = Conv2D(254, (1,1))(c5)
  p4 = upsample_add(p5, c4_lat)
  p3 = upsample_add(p4, c3_lat)
  p2 = upsample_add(p3, c2_lat)

  p4 = Conv2D(256, (3,3))(p4)
  p3 = Conv2D(256, (3,3))(p3)
  p2 = Conv2D(256, (3,3))(p2)

  # Segnet
  o = concatenate([p2, p3, p4, p5])
  # o = levels[ fpn_level ]

  o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
  o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
  o = ( BatchNormalization())(o)

  o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
  o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
  o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(o)
  o = ( BatchNormalization())(o)

  o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
  o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
  o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
  o = ( BatchNormalization())(o)

  o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
  o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
  o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
  o = ( BatchNormalization())(o)

  o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
  o_shape = Model(img_input , o ).output_shape
  outputHeight = o_shape[2]
  outputWidth = o_shape[3]

  o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
  o = (Permute((2, 1)))(o)
  o = (Activation('softmax'))(o)
  model = Model( img_input , o )
  model.outputWidth = outputWidth
  model.outputHeight = outputHeight

  return model
