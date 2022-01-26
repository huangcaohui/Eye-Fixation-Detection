# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers, Sequential


class LearningPrior(keras.Model):
    def __init__(self, shape, nb_gaussian, nb_filters_out, init=initializers.RandomNormal()):
        super(LearningPrior, self).__init__()

        self._nb_gaussian = nb_gaussian
        self._nb_filters_out = nb_filters_out
        self.height = shape[0]
        self.width = shape[1]
        self._init = init

        e = self.height / self.width
        e1 = (1 - e) / 2
        e2 = e1 + e

        # [h, w]
        x_t = tf.matmul(tf.ones((self.height, 1)), tf.expand_dims(tf.linspace(0.0, 1.0, self.width), 0))    # 行相同
        y_t = tf.matmul(tf.cast(tf.expand_dims(tf.linspace(e1, e2, self.height), 1), dtype='float32'), tf.ones((1, self.width)))  # 列相同

        # [h, w] => [h, w, 1] => [h, w, _nb_gaussian]
        self.x_t = tf.tile(tf.expand_dims(x_t, axis=-1), [1, 1, self._nb_gaussian])     # 在第三个维度上平铺
        self.y_t = tf.tile(tf.expand_dims(y_t, axis=-1), [1, 1, self._nb_gaussian])     # 在第三个维度上平铺

        self.layer = Sequential([layers.Conv2D(self._nb_filters_out, kernel_size=(5, 5), padding='same', dilation_rate=4),
                                 layers.Activation('relu')])

    def build(self, input_shape):
        # var为带更新变量，分别存储二维高斯分布的均值mu_x、mu_y和标准差sigma_x、sigma_y，[_nb_gaussian*4]
        self.var = self.add_weight('prior_var', shape=[self._nb_gaussian * 4], initializer=self._init, trainable=True)

        self.built = True

    def call(self, inputs, training=None, eps=np.finfo('float').eps):
        batchsz = tf.shape(inputs)[0]
        mu_x = self.var[:self._nb_gaussian]
        mu_y = self.var[self._nb_gaussian:self._nb_gaussian * 2]
        sigma_x = self.var[self._nb_gaussian * 2:self._nb_gaussian * 3]
        sigma_y = self.var[self._nb_gaussian * 3:]

        mu_x = tf.clip_by_value(mu_x, 0.25, 0.75)   # x均值限制在0.25~0.75之间
        mu_y = tf.clip_by_value(mu_y, 0.35, 0.65)   # y均值限制在0.35~0.65之间

        sigma_x = tf.clip_by_value(sigma_x, 0.1, 0.9)   # x方差限制在0.1~0.9之间
        sigma_y = tf.clip_by_value(sigma_y, 0.2, 0.8)   # y方差限制在0.2~0.8之间

        # 二维高斯函数，broadcasting自动对齐h、w维度， [h, w, _nb_gaussian]
        gaussian = 1 / (2*np.pi*sigma_x*sigma_y + eps) * \
                        tf.exp(-((self.x_t - mu_x)**2 / (2*sigma_x**2 + eps) +
                                 (self.y_t - mu_y)**2 / (2*sigma_y**2 + eps)))

        # 求每个二维高斯函数的最大值，axis维度保持为1，[h, w, _nb_gaussian] => [1, 1, _nb_gaussian]
        max_gauss = tf.reduce_max(gaussian, axis=[0, 1], keepdims=True)

        # 归一化
        gaussian = gaussian / max_gauss

        # 在batchsize维度平铺，[h, w, _nb_gaussian] => [1, h, w, _nb_gaussian] => [b, h, w, _nb_gaussian]
        output = tf.tile(tf.expand_dims(gaussian, axis=0), [batchsz, 1, 1, 1])

        # 在通道维度上将输入和先验图合并，[b, h, w, _nb_filters_out + _nb_gaussian]
        concate_net = tf.concat([inputs, output], axis=-1)
        # 卷积，[b, h, w, _nb_filters_out + _nb_gaussian] => [b, h, w, _nb_filters_out]
        learnded_priors = self.layer(concate_net, training=training)

        return learnded_priors


if __name__ == '__main__':
    shape = [180, 320]
    gaussian_prior = LearningPrior(shape, 16, 128)
    x = tf.random.truncated_normal([4]+shape+[128], dtype='float32')
    var = gaussian_prior(x)
    gaussian_prior.summary()

    print('output shape：', var.shape)
    print('End of program!')
