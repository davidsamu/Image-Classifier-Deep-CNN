#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully configurable Deep Convolutional Neural Network for Image Classification,
trained by Backpropagation. Implementation does not use of any Deep Learning
package (Tensorflow, PyTorch, etc), only Numpy.

@author: David Samu
"""

# %% Import

import os
import sys

import numpy as np


# Set this to your path to main folder!
proj_dir = '/home/david/Modelling/ImgClass'

sys.path.insert(1, proj_dir)

from img_class import cnn, utils

os.chdir(proj_dir)

# Image filename template
f_img = 'images/train-52x52'  # Set to your path to main folder of dataset!
f_img = os.path.join(proj_dir, f_img, '{0:d}', '{0:d}_{1:04d}.bmp')

# Result figure filename template
f_fig = os.path.join(proj_dir, 'results', '{}', '{}.png')


# %% Task params

# Number of image classes.
n_classes = 12

# Image params: number of color channels (RGB: 3, greyscale: 1), height and
# width, in pixel
img_pars = {'c': 3, 'h': 52, 'w': 52}


# %% Init CNN

# CNN hyperparams

# Each list item (row) is a NN layer (ordered from first to last), with a
# dictionary of type-specific parameters:

# - type: conv, relu, pool, full  (fully connected)
# - oD: output dimensions / number of features [all layer type]
# - F: filter size [conv and pool only]
# - S: stride [conv and pool only]

# Number, type and params of layers are fully configurable.

CNN = [{'type': 'conv', 'oD': 40, 'F': 5, 'S': 1},  # 1st / input layer
       {'type': 'relu'},
       {'type': 'pool', 'F': 2, 'S': 2},
#       {'type': 'conv', 'oD': 30, 'F': 3, 'S': 1},  # add / uncomment lines
#       {'type': 'relu'},                            # to use more layers
#       {'type': 'pool', 'F': 2, 'S': 2},
#       {'type': 'conv', 'oD': 20, 'F': 3, 'S': 1},
#       {'type': 'relu'},
#       {'type': 'pool', 'F': 2, 'S': 2},
       {'type': 'full', 'oD': 12}]                  # last / output layer


# Init CNN (weights, biases & misc fields)
cnn.init_network(CNN, img_pars)


# Print some info on network architecture to console
print('\nCNN architecture:\n')
for i in range(len(CNN)):
    print(CNN[i]['name'], CNN[i]['type'], CNN[i]['iH'], CNN[i]['iW'],
          CNN[i]['F'], CNN[i]['iD'], CNN[i]['oD'])


# %% Training CNN

# Number of images to use from full image set (to speed up training & testing)
n_train = 2000
train_set = list(range(n_train))

# Training params
n_batches = 2000   # number of batches

lr0 = 0.002          # initial learning rate
lr_drop = 0.5        # step decay drop of learning rate
lr_drop_time = 500   # number of batches between drops
gamma = 0.8          # momentum strength (0: momentum off)


# Objects to store training results
loss = np.zeros([n_batches, n_classes])
acc = np.zeros([n_batches, n_classes])

# Do training
print('\nStarting training...\n')
for ibatch in range(n_batches):

    # learning rate update according to decay schedule
    lr = lr0 * lr_drop**int(ibatch / lr_drop_time)

    for iclass in range(n_classes):

        # import
        iimg = np.random.choice(train_set)
        img = utils.read_img(f_img.format(iclass+1, iimg))

        # forward pass
        cnn.forward_pass(img, CNN)

        # loss
        loss[ibatch, iclass] = cnn.cross_entropy_loss(CNN, iclass)

        # accuracy
        acc[ibatch, iclass] = (iclass == cnn.ML_choice(CNN))

        # backward pass
        cnn.backprop(img, CNN, iclass)

        # weight update
        cnn.weight_update(CNN, lr, gamma)

    # report progress
    utils.report_learning_progress(ibatch, loss, acc)

print('\nEnd training\n')

# Plot trainig progress
fname = f_fig.format(len(CNN), 'train')
utils.plot_learning_progress(loss, acc, fname)


# %% Testing CNN

# Params
n_test = 100
test_set = list(range(n_train, n_train+n_test))

# Objects to store test results
acc_test = np.zeros([n_test, n_classes])

# Do testing
print('\nStarting testing...\n')
for iclass in range(n_classes):
    print('\t{}'.format(iclass))

    for i, iimg in enumerate(test_set):

        img = utils.read_img(f_img.format(iclass+1, iimg))      # import
        cnn.forward_pass(img, CNN)                              # forward pass
        acc_test[i, iclass] = (iclass == cnn.ML_choice(CNN))    # accuracy

print('\nEnd testing...\n')

# Plot test results
fname = f_fig.format(len(CNN), 'test')
utils.plot_test_accuracy(acc_test, fname)
