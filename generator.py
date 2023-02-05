import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

class Generator():
    def __init__(self):
        self.model = ""

    def make_generator_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Reshape((7, 7, 256)))
        assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)

        return self.model

    def generator_loss(self, fake_output):
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        return cross_entropy(tf.ones_like(fake_output), fake_output)