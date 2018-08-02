import os
import random
import matplotlib.pyplot as plt
import sys
import math
import tifffile as tiff
import numpy as np
from PIL import Image
from libtiff import TIFF as LIBLIFF

# Parameter

input_rows, input_cols = 256, 256
step_pixel_size = 50
data_path = "/mnt/local/zongwei/dataset/EM/" # Mayo Machine
train_idx = [n for n in range(0, 24)]
valid_idx = [n for n in range(24,  27)]
test_idx = [n for n in range(27, 30)]
batch_size = 12
lr = 1e-3
GPU_COUNT = 1
nb_epoch = 100
patience = 10
data_augmentation = False

# multiple GPUs

import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM


class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # If outputs are numbers without dimensions, add a batch dim.
                def add_dim(tensor):
                    """Add a dimension to tensors that don't have any."""
                    if K.int_shape(tensor) == ():
                        return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
                    return tensor
                outputs = list(map(add_dim, outputs))

                # Concatenate
                merged.append(KL.Concatenate(axis=0, name=name)(outputs))
        return merged

# custom layers

from keras.engine import Layer, InputSpec
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K

class Scale(Layer):
    
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        #self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        #self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Models

import keras
from model import U_Net, U_ConvNet, Nest_Net, Nesen_Net
from model import mean_iou, bce_dice_loss, dice_coef
import tensorflow as tf
def prepare_model(architecture, input_rows, input_cols):

    # U_Net, U_Inception, U_ResNet, U_DenseNet
    # Nest_Net, Nest_Inception, Nest_ResNet, Nest_DenseNet
    # Fully_Nest_Net, Fully_Nest_Inception, Fully_Nest_ResNet, Fully_Nest_DenseNet

    with tf.device("/gpu:0"):
        if architecture == "U_Net":
            model = U_Net(input_rows, input_cols, 1)
        elif architecture == "Nest_Net":
            model = Nest_Net(input_rows, input_cols, 1)
        elif architecture == "Nesen_Net":
            model = Nesen_Net(input_rows, input_cols, 1)
        else:
            print("No matching model architecture.")

    if GPU_COUNT > 1:
        model = ParallelModel(model, GPU_COUNT)

    return model

def compile_model(model):
    #optimizer = keras.optimizers.RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
    #optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[mean_iou, dice_coef])

    return model


# Data

from augmentation import random_zoom, elastic_transform, random_rotation
def aug_data(orgimgs, orgmask):

    mask = []
    imgs = []

    for i in range( orgimgs.shape[0] ):

        imgs_ = orgimgs[i]
        mask_ = orgmask[i]

        mask.append(mask_)
        imgs.append(imgs_)

        for _ in range(1):
            mask.append(np.flipud(mask_))
            imgs.append(np.flipud(imgs_))

        for _ in range(1):
            mask.append(np.fliplr(mask_))
            imgs.append(np.fliplr(imgs_))

        for _ in range(1):
            mask.append(np.fliplr(np.flipud(mask_)))
            imgs.append(np.fliplr(np.flipud(imgs_)))

        for _ in range(1):
            _x, _y = random_zoom(imgs_, mask_, (0.9, 1.1))
            mask.append(_y)
            imgs.append(_x)

        for _ in range(1):
            _x, _y = random_rotation(imgs_, mask_, 10)
            mask.append(_y)
            imgs.append(_x)

    imgs = np.array(imgs)
    mask = np.array(mask)

    return imgs, mask

def load_data(idx):

    mask = tiff.imread( data_path + 'train-labels.tif' )
    imgs = tiff.imread( data_path + 'train-volume.tif' )
    mask = np.array(mask)
    imgs = np.array(imgs)

    # Get and resize train images and masks
    X_train = []
    Y_train = []
    sys.stdout.flush()
    for i in idx:
        im = np.squeeze(imgs[i])
        mk = np.squeeze(mask[i])
        im = np.array(im)
        mk = np.array(mk)

        rows, cols = im.shape[0], im.shape[1]
        nb_row = int( math.floor( (rows-input_rows) / step_pixel_size) ) + 1
        nb_cols = int( math.floor( (cols-input_cols) / step_pixel_size) ) + 1
        row_list = [x*step_pixel_size for x in range(nb_row)]
        col_list = [x*step_pixel_size for x in range(nb_cols)]

        for i in row_list:
            for j in col_list:
                X_train.append( im[i:i+input_rows, j:j+input_cols] )
                Y_train.append( mk[i:i+input_rows, j:j+input_cols] )

                X_train.append( im[rows-input_rows:rows, j:j+input_cols] )
                Y_train.append( mk[rows-input_rows:rows, j:j+input_cols] )

            X_train.append( im[i:i+input_rows, cols-input_cols:cols] )
            Y_train.append( mk[i:i+input_rows, cols-input_cols:cols] )

        X_train.append( im[rows-input_rows:rows, cols-input_cols:cols] )
        Y_train.append( mk[rows-input_rows:rows, cols-input_cols:cols] )

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = np.expand_dims(X_train,  axis=-1)
    Y_train = np.expand_dims(Y_train,  axis=-1)

    return X_train, Y_train

def test_single_image(im, model):
    # im = (96,96), 0-255
    im = np.expand_dims(im,  axis=-1)
    im = np.expand_dims(im,  axis=0)
    im = im.astype('float32')
    im /= 255.0
    
    return np.squeeze(model.predict(im, verbose=0))
    
def test_data(idx, model):

    mask = tiff.imread( data_path + 'train-labels.tif' )
    imgs = tiff.imread( data_path + 'train-volume.tif' )
    mask = np.array(mask)
    imgs = np.array(imgs)

    # Get and resize train images and masks
    x_test = []
    y_test = []
    p_test = []
    sys.stdout.flush()
    for i in idx:
        im = np.squeeze(imgs[i])
        mk = np.squeeze(mask[i])
        im = np.array(im)
        mk = np.array(mk)
        x_test.append(im)
        y_test.append(mk)
        
        x_patch = []
        n_orgim = np.zeros(mk.shape, dtype="int16")
        p_orgim = np.zeros(mk.shape, dtype="float")

        rows, cols = im.shape[0], im.shape[1]
        nb_row = int( math.floor( (rows-input_rows) / step_pixel_size) ) + 1
        nb_cols = int( math.floor( (cols-input_cols) / step_pixel_size) ) + 1
        row_list = [x*step_pixel_size for x in range(nb_row)]
        col_list = [x*step_pixel_size for x in range(nb_cols)]
        
        for i in row_list:
            for j in col_list:
                p_orgim[i:i+input_rows, j:j+input_cols] = p_orgim[i:i+input_rows, j:j+input_cols] + test_single_image(im[i:i+input_rows, j:j+input_cols], model)
                n_orgim[i:i+input_rows, j:j+input_cols] = n_orgim[i:i+input_rows, j:j+input_cols] + 1
                p_orgim[rows-input_rows:rows, j:j+input_cols] = p_orgim[rows-input_rows:rows, j:j+input_cols] + test_single_image(im[rows-input_rows:rows, j:j+input_cols], model)
                n_orgim[rows-input_rows:rows, j:j+input_cols] = n_orgim[rows-input_rows:rows, j:j+input_cols] + 1
            p_orgim[i:i+input_rows, cols-input_cols:cols] = p_orgim[i:i+input_rows, cols-input_cols:cols] + test_single_image(im[i:i+input_rows, cols-input_cols:cols], model)
            n_orgim[i:i+input_rows, cols-input_cols:cols] = n_orgim[i:i+input_rows, cols-input_cols:cols] + 1
        p_orgim[rows-input_rows:rows, cols-input_cols:cols] = p_orgim[rows-input_rows:rows, cols-input_cols:cols] + test_single_image(im[rows-input_rows:rows, cols-input_cols:cols], model)
        n_orgim[rows-input_rows:rows, cols-input_cols:cols] = n_orgim[rows-input_rows:rows, cols-input_cols:cols] + 1
        
        
        p_orgim = p_orgim / n_orgim
        p_test.append(p_orgim)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    p_test = np.array(p_test)

    return x_test, y_test, p_test

def public_test(model):
    
    imgs = tiff.imread( data_path + 'test-volume.tif' )
    imgs = np.array(imgs)

    # Get and resize train images and masks
    x_test = []
    p_test = []
    sys.stdout.flush()
    for i in range(imgs.shape[0]):
        im = np.squeeze(imgs[i])
        im = np.array(im)
        x_test.append(im)
        
        x_patch = []
        n_orgim = np.zeros(im.shape, dtype="int16")
        p_orgim = np.zeros(im.shape, dtype="float")

        rows, cols = im.shape[0], im.shape[1]
        nb_row = int( math.floor( (rows-input_rows) / step_pixel_size) ) + 1
        nb_cols = int( math.floor( (cols-input_cols) / step_pixel_size) ) + 1
        row_list = [x*step_pixel_size for x in range(nb_row)]
        col_list = [x*step_pixel_size for x in range(nb_cols)]
        
        for i in row_list:
            for j in col_list:
                p_orgim[i:i+input_rows, j:j+input_cols] = p_orgim[i:i+input_rows, j:j+input_cols] + test_single_image(im[i:i+input_rows, j:j+input_cols], model)
                n_orgim[i:i+input_rows, j:j+input_cols] = n_orgim[i:i+input_rows, j:j+input_cols] + 1
                p_orgim[rows-input_rows:rows, j:j+input_cols] = p_orgim[rows-input_rows:rows, j:j+input_cols] + test_single_image(im[rows-input_rows:rows, j:j+input_cols], model)
                n_orgim[rows-input_rows:rows, j:j+input_cols] = n_orgim[rows-input_rows:rows, j:j+input_cols] + 1
            p_orgim[i:i+input_rows, cols-input_cols:cols] = p_orgim[i:i+input_rows, cols-input_cols:cols] + test_single_image(im[i:i+input_rows, cols-input_cols:cols], model)
            n_orgim[i:i+input_rows, cols-input_cols:cols] = n_orgim[i:i+input_rows, cols-input_cols:cols] + 1
        p_orgim[rows-input_rows:rows, cols-input_cols:cols] = p_orgim[rows-input_rows:rows, cols-input_cols:cols] + test_single_image(im[rows-input_rows:rows, cols-input_cols:cols], model)
        n_orgim[rows-input_rows:rows, cols-input_cols:cols] = n_orgim[rows-input_rows:rows, cols-input_cols:cols] + 1
        
        
        p_orgim = p_orgim / n_orgim
        p_test.append(p_orgim)

    x_test = np.array(x_test)
    p_test = np.array(p_test)

    return x_test, p_test

# Quality

def compute_iou(img1, img2):

    img1 = np.array(img1)
    img2 = np.array(img2)

    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    IoU = np.zeros( (img1.shape[0],), dtype=np.float32)
    for i in range(img1.shape[0]):
        im1 = np.squeeze(img1[i]>0.5)
        im2 = np.squeeze(img2[i]>0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        IoU[i] = 2. * intersection.sum() * 100.0 / (im1.sum() + im2.sum())
        #database.display_image_mask_pairs(im1, im2)

    return IoU

def plot_feature_map(layer, model, image):
    
    fm = {}
    for i in range(len(layer)):
        layer_name = layer[i]
        intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(image)
        fm[layer[i]] = intermediate_output  
    return fm