#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Image Classification project.

@author: David Samu
"""

import os

import numpy as np
import imageio

import matplotlib.pyplot as plt


# %% Image I/O and preprocessing

def read_img(img_path, norm=True):
    """Read in and pre-process image."""

    img = imageio.imread(img_path)

    if norm:
        img = norm_img(img)

    return img


def norm_img(img, rng=255):
    """Normalize image of given range."""
    return (img-rng/2) / rng


def report_learning_progress(ibatch, loss, acc):
    """Report learning progress to console."""

    mloss = loss[ibatch, :].mean()
    macc = acc[ibatch, :].mean()
    sbatch = str(ibatch).rjust(4)
    print('\tbatch: {} | loss: {:.3f} | acc: {:.3f}'.format(sbatch,
                                                            mloss, macc))


# %% Misc math

def softmax(v):
    """Return soft-max of vector."""

    exps = np.exp(v-v.max())
    return exps / np.sum(exps)


def max_idx(a):
    """Indexes of the maximal elements of a N-dimensional array."""

    return np.unravel_index(np.argmax(a, axis=None), a.shape)


# %% Plotting

def plot_learning_progress(loss, acc, fname=None):
    """Plot learning progress (loss and accuracy)."""

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    mloss = loss.mean(1)
    macc = acc.mean(1)

    # Loss
    ax1.plot(mloss)
    ax1.set_ylabel('loss', color='blue')
    ax1.tick_params('y', colors='blue')

    # Accuracy
    ax2 = ax1.twinx()
    ax2.plot(macc, 'orange')
    ax2.set_ylabel('accuracy', color='orange')
    ax2.tick_params('y', colors='orange')

    ax1.set_xlabel('batch number')
    ax1.set_title('Training loss & accuracy')

    if fname is not None:
        save_fig(fname, fig)


def plot_test_accuracy(acc_test, fname=None):
    """Plot test accuracy per class."""

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    lbls = np.array(range(acc_test.shape[1])) + 1
    ax.bar(lbls, 100 * acc_test.mean(0))

    ax.set_xticks(lbls)
    plt.ylim([0, 100])

    ttl = 'Test accuracy, grand mean: {:.1f}'.format(100*acc_test.mean())
    ax.set_title(ttl)
    ax.set_xlabel('class index')
    ax.set_ylabel('mean accuracy (%)')

    if fname is not None:
        save_fig(fname, fig)


def create_dir(f):
    """Create directory if it does not already exist."""

    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return


def save_fig(ffig, fig=None, dpi=180, close=True, tight_layout=True):
    """Save composite (GridSpec) figure to file."""

    # Init figure and folder to save figure into.
    create_dir(ffig)

    if fig is None:
        fig = plt.gcf()

    if tight_layout:
        fig.tight_layout()

    fig.savefig(ffig, dpi=dpi, bbox_inches='tight')

    if close:
        fig.clf()
        plt.close(fig)
