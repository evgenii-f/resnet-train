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


class Custom_Sigmoid(Layer):

    def __init__(self, c1=-1., trainable=False):
        super(Custom_Sigmoid, self).__init__()
        self.supports_masking = True
        self.c1 = c1
        self.trainable = trainable
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "c1": self.c1,
        })
        return config

    def build(self, input_shape):
        self.c1_factor = K.variable(self.c1,
                                      dtype=K.floatx(),
                                      name='c1_factor')
        if self.trainable:
            self._trainable_weights.append(self.c1_factor)

    def call(self, inputs, mask=None):
        return tensorflow.math.divide(1., tensorflow.math.add(1., tensorflow.math.exp(tensorflow.math.multiply(self.c1_factor, inputs))))


def squeeze_excite_block(input_tensor, nb_filter, ratio=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(nb_filter//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(nb_filter, kernel_initializer='he_normal', use_bias=False)(x)
    x = Custom_Sigmoid(c1=-1.)(x)
    result = multiply([input_tensor, x])
    return result

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

def residual_se_block(input_tensor, nb_filter):
    out = Conv2D(nb_filter, (3,3), kernel_initializer = 'he_normal', padding='same')(input_tensor)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = Conv2D(nb_filter, (3,3), kernel_initializer = 'he_normal', padding='same')(out)
    out = BatchNormalization()(out)
    #out = ReLU()(out)
    se = squeeze_excite_block(out, nb_filter)
    
    x_skip = Conv2D(nb_filter,(1,1), kernel_initializer = 'he_normal', padding='same')(input_tensor)
    # x_skip = BatchNormalization()(x_skip) # hier oder nach skip

    x = add([se, x_skip])
    x = ReLU()(x) # hier oder nach skip
    return x

def my_vers_1_res(img_rows=512, img_cols=512):

    nb_filter = [32,64,128,256,512,512]
    nb_filter_deep = [32,32]
    bn_axis = 3
    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(img_rows, img_cols, 3), name='main_input')
    
    conv0_1 = residual_block(img_input, nb_filter=nb_filter_deep[0])
    conv0_1_se = residual_se_block(conv0_1, nb_filter=nb_filter_deep[0])
    pool0 = MaxPooling2D((2, 2), strides=(2, 2), name='pool0')(conv0_1) # mein Conv2D

    conv1_1 = residual_block(pool0, nb_filter=nb_filter[1])
    conv1_1_se = residual_se_block(conv1_1, nb_filter=nb_filter[1])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1) # mein Conv2D
    
    up1_1 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up11', padding='same')(conv1_1) #vergrößert!
    up1_1 = BatchNormalization()(up1_1)
    up1_1 = ReLU()(up1_1)
    
    conv0_2 = concatenate([up1_1, conv0_1], name='merge02', axis=bn_axis)
    conv0_2_se = residual_se_block(conv0_2, nb_filter=nb_filter[0])
    
    ########################################################################################################
    conv2_1 = residual_block(pool1, nb_filter=nb_filter[2])
    conv2_1_se = residual_se_block(pool1, nb_filter=nb_filter[2])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1) #vergrößert!
    up1_2 = BatchNormalization()(up1_2)
    up1_2 = ReLU()(up1_2)

    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2_se = residual_se_block(conv1_2, nb_filter=nb_filter[1])
#############################################################################################################################
    conv3_1 = residual_block(pool2, nb_filter=nb_filter[3])
    conv3_1_se = residual_se_block(conv3_1, nb_filter=nb_filter[3]) 
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    up2_2 = BatchNormalization()(up2_2)
    up2_2 = ReLU()(up2_2)
    
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2_se = residual_se_block(conv2_2, nb_filter=nb_filter[2])

    conv4_1 = residual_block(pool3, nb_filter=nb_filter[4])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    up3_2 = BatchNormalization()(up3_2)
    up3_2 = ReLU()(up3_2)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2_se = residual_se_block(conv3_2, nb_filter=nb_filter[3])

    conv5_1 = residual_block(pool4, nb_filter=nb_filter[5])

    up4_2 = Conv2DTranspose(nb_filter[4], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    up4_2 = BatchNormalization()(up4_2)
    up4_2 = ReLU()(up4_2)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2_se = residual_se_block(conv4_2, nb_filter=nb_filter[4])

    up3_3 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2_se)
    up3_3 = BatchNormalization()(up3_3)
    up3_3 = ReLU()(up3_3)
    conv3_3 = concatenate([up3_3, conv3_1_se, conv3_2_se], name='merge33', axis=bn_axis)
    conv3_3 = residual_block(conv3_3, nb_filter=nb_filter[3])

    up2_4 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    up2_4 = BatchNormalization()(up2_4)
    up2_4 = ReLU()(up2_4)
    conv2_4 = concatenate([up2_4, conv2_1_se, conv2_2_se], name='merge24', axis=bn_axis)
    conv2_4 = residual_block(conv2_4, nb_filter=nb_filter[2])
    

    up1_5 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    up1_5 = BatchNormalization()(up1_5)
    up1_5 = ReLU()(up1_5)
    conv1_5 = concatenate([up1_5, conv1_1_se, conv1_2_se], name='merge15', axis=bn_axis)
    conv1_5 = residual_block(conv1_5, nb_filter=nb_filter[1])

    up0_6 = Conv2DTranspose(nb_filter_deep[0], (2, 2), strides=(2, 2), name='up06', padding='same')(conv1_5)
    up0_6 = BatchNormalization()(up0_6)
    up0_6 = ReLU()(up0_6)
    conv0_6 = concatenate([up0_6, conv0_1_se, conv0_2_se], name='merge06', axis=bn_axis)
    conv0_6 = residual_block(conv0_6, nb_filter=nb_filter[0])

    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv0_6)

    model = Model(inputs=img_input, outputs=nestnet_output_4)

    return model


def res_classificator(n_class, img_rows=512, img_cols=512):
    nb_filter = [32,64,128,256,512,512]
    nb_filter_deep = [32,32]
    bn_axis = 3
    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(img_rows, img_cols, 3), name='main_input')
    
    conv0_1 = residual_block(img_input, nb_filter=nb_filter_deep[0])
    conv0_1_se = residual_se_block(conv0_1, nb_filter=nb_filter_deep[0])
    pool0 = MaxPooling2D((2, 2), strides=(2, 2), name='pool0')(conv0_1) # mein Conv2D

    conv1_1 = residual_block(pool0, nb_filter=nb_filter[1])
    conv1_1_se = residual_se_block(conv1_1, nb_filter=nb_filter[1])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1) # mein Conv2D
    
    up1_1 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up11', padding='same')(conv1_1) #vergrößert!
    up1_1 = BatchNormalization()(up1_1)
    up1_1 = ReLU()(up1_1)
    
    conv0_2 = concatenate([up1_1, conv0_1], name='merge02', axis=bn_axis)
    conv0_2_se = residual_se_block(conv0_2, nb_filter=nb_filter[0])
    
    ########################################################################################################
    conv2_1 = residual_block(pool1, nb_filter=nb_filter[2])
    conv2_1_se = residual_se_block(pool1, nb_filter=nb_filter[2])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1) #vergrößert!
    up1_2 = BatchNormalization()(up1_2)
    up1_2 = ReLU()(up1_2)

    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2_se = residual_se_block(conv1_2, nb_filter=nb_filter[1])
#############################################################################################################################
    conv3_1 = residual_block(pool2, nb_filter=nb_filter[3])
    conv3_1_se = residual_se_block(conv3_1, nb_filter=nb_filter[3]) 
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    up2_2 = BatchNormalization()(up2_2)
    up2_2 = ReLU()(up2_2)
    
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2_se = residual_se_block(conv2_2, nb_filter=nb_filter[2])

    conv4_1 = residual_block(pool3, nb_filter=nb_filter[4])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    up3_2 = BatchNormalization()(up3_2)
    up3_2 = ReLU()(up3_2)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2_se = residual_se_block(conv3_2, nb_filter=nb_filter[3])

    conv5_1 = residual_block(pool4, nb_filter=nb_filter[5])

    # my changes are from here:
    x = GlobalAveragePooling2D()(conv5_1)
    x = Flatten()(x)
    x = Dense(n_class, activation='softmax', name='output', kernel_initializer='he_normal')(x) #multi-class

    model = Model(inputs=img_input, outputs=x)
    return model
    
def vers1_classificator(n_class, img_rows=512, img_cols=512):

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

    x = GlobalAveragePooling2D()(conv5_1)
    x = Flatten()(x)
    x = Dense(n_class, activation='softmax', name='output', kernel_initializer='he_normal')(x) #multi-class

    model = Model(inputs=img_input, outputs=x)
    return model