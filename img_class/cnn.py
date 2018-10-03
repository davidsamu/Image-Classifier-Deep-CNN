#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN functions for image classification.

@author: David Samu
"""


import numpy as np

from img_class import utils


# %% Initialization

def init_network(CNN, img_pars, std=0.01, bias=0):
    """Extend hyper-params of CNN."""

    for i in range(len(CNN)):

        CNN[i]['name'] = 'layer {}'.format(i+1)  # layer name

        # Input dims
        CNN[i]['iD'] = CNN[i-1]['oD'] if i != 0 else img_pars['c']
        CNN[i]['iH'] = CNN[i-1]['oH'] if i != 0 else img_pars['h']
        CNN[i]['iW'] = CNN[i-1]['oW'] if i != 0 else img_pars['w']

        # ReLu layer
        if CNN[i]['type'] == 'relu':
            CNN[i]['S'] = 1              # stride = 1
            CNN[i]['F'] = 1              # filter size = 1
            CNN[i]['oD'] = CNN[i]['iD']  # out dim = input dim

        # Pooling layer
        if CNN[i]['type'] == 'pool':
            CNN[i]['oD'] = CNN[i]['iD']  # out dim = input dim

        # Fully connected layer
        if CNN[i]['type'] == 'full':
            CNN[i]['S'] = 1              # stride = 1
            CNN[i]['F'] = CNN[i]['iH']   # filter size = input size (iH == iW)

        # Index vectors (addressing positions of convolution / pooling window)
        CNN[i]['ixh'] = np.arange(0, CNN[i]['iH']+1-CNN[i]['F'], CNN[i]['S'])
        CNN[i]['ixw'] = np.arange(0, CNN[i]['iW']+1-CNN[i]['F'], CNN[i]['S'])

        # Output dims
        CNN[i]['oH'] = len(CNN[i]['ixh'])
        CNN[i]['oW'] = len(CNN[i]['ixw'])

        # Weights, biases and their batch gradients
        if CNN[i]['type'] in ['conv', 'full']:

            w_shape = (CNN[i]['F'], CNN[i]['F'], CNN[i]['iD'], CNN[i]['oD'])
            CNN[i]['W'] = np.random.normal(0, std, w_shape).squeeze()
            CNN[i]['b'] = bias * np.ones(CNN[i]['oD'])

            CNN[i]['dW'] = []  # lists to store gradients for each batch item
            CNN[i]['db'] = []

            CNN[i]['vW'] = 0  # momentum accumulated gradients
            CNN[i]['vb'] = 0

        # Activation
        CNN[i]['x'] = np.zeros((CNN[i]['oH'], CNN[i]['oW'], CNN[i]['oD']))
        CNN[i]['x'] = CNN[i]['x'].squeeze()


# %% Forward pass

def forward_pass_layer(inp, CNNi):
    """Pass input through a layer."""

    # ReLu layer
    if CNNi['type'] == 'relu':
        CNNi['x'] = np.maximum(inp, 0)
        return

    f = CNNi['F']

    # Pool layer: slide window along input and max-pool under it
    if CNNi['type'] == 'pool':
        for ho, hi in enumerate(CNNi['ixh']):
            for wo, wi in enumerate(CNNi['ixw']):
                CNNi['x'][ho, wo] = np.amax(inp[hi:hi+f, wi:wi+f], (0, 1))
        return

    W, b = CNNi['W'], CNNi['b']
    tp_ax = 2 * [range(inp.ndim)]  # axes to perform tensor product along

    # Fully connected layer: single tensor product
    if CNNi['type'] == 'full':
        CNNi['x'] = np.tensordot(inp, W, tp_ax) + b
        return

    # Conv layer: slide filter along input and convolve
    if CNNi['type'] == 'conv':
        for ho, hi in enumerate(CNNi['ixh']):
            for wo, wi in enumerate(CNNi['ixw']):
                CNNi['x'][ho, wo] = np.tensordot(inp[hi:hi+f, wi:wi+f],
                                                 W, tp_ax) + b
        return


def forward_pass(img, CNN):
    """Pass image through network."""

    # Going foward from first to last layer
    for i in range(len(CNN)):

        if i == 0:   # first layer
            forward_pass_layer(img, CNN[i])
        else:        # subsequent layers
            forward_pass_layer(CNN[i-1]['x'], CNN[i])


# %% Loss function and ML choice

def cross_entropy_loss(CNN, ilbl):
    """Calculate soft-max + cross entropy loss function."""

    # Final layer activity
    y = CNN[-1]['x'].squeeze()

    # Softmax final layer
    smax = utils.softmax(y)

    # Cross-entropy on one-hot encoded target label
    xe_loss = -np.log(smax[ilbl])

    return xe_loss


def ML_choice(CNN):
    """Return Maximum Likelihood choice to test accuracy."""

    y = CNN[-1]['x'].squeeze()
    ml_choice = np.argmax(y)

    return ml_choice


# %% Gradient backpropagation

def backprop_layer(inp, CNNi, delta):
    """Backpropagate gradients by one layer."""

    if CNNi['type'] == 'relu':

        # Gradient is cut at negative values.
        delta_prev = delta.copy()
        delta_prev[inp <= 0] = 0

    if CNNi['type'] == 'pool':

        # Gradient is 'routed' back to maximum value under pooling window.
        f = CNNi['F']
        delta_prev = np.zeros_like(inp)
        for ho, hi in enumerate(CNNi['ixh']):
            for wo, wi in enumerate(CNNi['ixw']):
                for di in range(delta.shape[-1]):
                    imaxh, imaxw = utils.max_idx(inp[hi:hi+f, wi:wi+f, di])
                    delta_prev[hi+imaxh, wi+imaxw, di] += delta[ho, wo, di]

    if CNNi['type'] == 'conv':

        # Init
        f, W = CNNi['F'], CNNi['W']
        iH, iW = CNNi['iH'], CNNi['iW']
        dW = np.zeros_like(CNNi['W'])
        delta_prev = np.zeros_like(inp)

        # Gradient of loss by W filters and biases
        tp_ax = [(0, 1), (0, 1)]  # axes to perform tensor product along
        for fi in range(f):
            for fj in range(f):
                ih = iH-f+fi+1  # end indexes of windows to use
                iw = iW-f+fj+1
                dW[fi, fj] = np.tensordot(inp[fi:ih, fj:iw], delta, tp_ax)

        CNNi['dW'].append(dW)
        CNNi['db'].append(np.sum(delta, (0, 1)))

        # Gradient of loss by input
        for ho, hi in enumerate(CNNi['ixh']):
            for wo, wi in enumerate(CNNi['ixw']):
                delta_prev[hi:hi+f, wi:wi+f] += np.sum(W * delta[ho, wo], -1)

    if CNNi['type'] == 'full':

        # Standard MPL backprop
        CNNi['dW'].append(np.multiply.outer(inp, delta))
        CNNi['db'].append(delta.copy())
        delta_prev = np.matmul(CNNi['W'], delta)

    return delta_prev


def backprop(img, CNN, ilbl):
    """Calculate gradients by BP."""

    # Derivative of cross-entropy loss by final SoftMax layer
    y = CNN[-1]['x'].squeeze()
    delta = utils.softmax(y)
    delta[ilbl] -= 1

    # Going backwards from last to first layer
    for i in range(len(CNN))[::-1]:

        if i > 0:  # second to last layers
            delta = backprop_layer(CNN[i-1]['x'], CNN[i], delta)

        else:      # first layer
            delta = backprop_layer(img, CNN[i], delta)


# %% Weight update

def weight_update(CNN, lr, gamma):
    """Perform weight and bias update."""

    for i in range(len(CNN)):
        if CNN[i]['type'] in ['conv', 'full']:

            # Average gradients in batch
            mean_dW = np.mean(np.array(CNN[i]['dW']), 0)
            mean_db = np.mean(np.array(CNN[i]['db']), 0)

            # Momentum update
            CNN[i]['vW'] = gamma * CNN[i]['vW'] + lr * mean_dW
            CNN[i]['vb'] = gamma * CNN[i]['vb'] + lr * mean_db

            # Parameter update
            CNN[i]['W'] -= CNN[i]['vW']
            CNN[i]['b'] -= CNN[i]['vb']

            # Reset gradient lists
            CNN[i]['dW'] = []
            CNN[i]['db'] = []
