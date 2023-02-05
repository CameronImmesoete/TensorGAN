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

class Dataset:
    def __init__(self, train_images="", train_dataset="", BUFFER_SIZE=60000, BATCH_SIZE=256):
        self.train_images = train_images
        self.train_dataset = train_dataset
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def importExample(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

        self.train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        self.train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
