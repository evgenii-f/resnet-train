import numpy as np
import os
import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import random
from PIL import Image
import sys
import time
np.set_printoptions(threshold=sys.maxsize)

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def residual_block(input, nb_filter):
    out = Conv2D(nb_filter, (3,3), kernel_initializer = 'he_normal', padding='same')(input)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = Conv2D(nb_filter, (3,3), kernel_initializer = 'he_normal', padding='same')(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    x_skip = Conv2D(nb_filter,(1,1), kernel_initializer = 'he_normal', padding='same')(input)
    x = add([x_skip, out])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def my_vers_1_res(img_rows=512, img_cols=512):

    nb_filter = [32,64,128,256,512,512]
    nb_filter_deep = [32,32]
    bn_axis = 3
    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(img_rows, img_cols, 3), name='main_input')
    
    conv0_1 = residual_block(img_input, nb_filter=nb_filter_deep[0], name='conv0_1')
    pool0 = MaxPooling2D((2, 2), strides=(2, 2), name='pool0')(conv0_1) # mein Conv2D

    conv1_1 = residual_block(pool0, nb_filter=nb_filter[1], name='conv1_1')
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1) # mein Conv2D
    
    ########################################################################################################
    conv2_1 = residual_block(pool1, nb_filter=nb_filter[2], name='conv2_1')
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

#############################################################################################################################
    conv3_1 = residual_block(pool2, nb_filter=nb_filter[3], name='conv3_1')
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = residual_block(pool3, nb_filter=nb_filter[4], name='conv4_1')
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = residual_block(pool4, nb_filter=nb_filter[5], name='conv5_1')

    model = Model(inputs=img_input, outputs=conv5_1)

    return model
