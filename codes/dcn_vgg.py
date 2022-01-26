# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


class DcnVgg(keras.Model):
    def __init__(self, nb_vgg_out):
        super(DcnVgg, self).__init__()

        self._nb_vgg_out = nb_vgg_out

        conv_layers = [
            # conv_1
            layers.Conv2D(64, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

            # conv_2
            layers.Conv2D(128, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

            # conv_3
            layers.Conv2D(256, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

            # conv_4
            layers.Conv2D(512, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, kernel_size=[3, 3], padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same'),

            layers.Conv2D(self._nb_vgg_out, kernel_size=[3, 3], padding='same', dilation_rate=2),
            layers.Activation('relu'),
            layers.Conv2D(self._nb_vgg_out, kernel_size=[3, 3], padding='same', dilation_rate=2),
            layers.Activation('relu'),
            layers.Conv2D(self._nb_vgg_out, kernel_size=[3, 3], padding='same', dilation_rate=2),
            layers.Activation('relu'),
        ]

        self.dcn_vgg = Sequential(conv_layers)

    def call(self, inputs, training=None):

        # [b, h, w, c] => [b, h/8, w/8, 512]
        out = self.dcn_vgg(inputs, training=training)

        return out


if __name__ == '__main__':
    shape = [360, 480]
    dcnvgg = DcnVgg(512)
    inputs = tf.random.truncated_normal([4]+shape+[3], dtype='float32')  # [b, h, w, c]
    out = dcnvgg(inputs)
    dcnvgg.summary()

    print('shape of out:', out.shape)
    print('End of program!')
