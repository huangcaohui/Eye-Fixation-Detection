# -*- coding:utf-8 -*-

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def kld(y_true, y_pred, eps=np.finfo('float').eps):
    """
        KL散度（Kullback-Leibler divergence）， 计算实际显著性图和预测值之间的KL散度
        首先将每个图像的值除以它们的总和，以产生和为1的分布。

    Args:
        y_true (tensor, float32): 4-d的真实显著性图，值介于0~1之间
        y_pred (tensor, float32): 4-d的预测显著性图，值介于0~1之间
        eps (scalar, float, optional): 小数值，避免数值不稳定

    Returns:
        tensor, float32: 关于误差的0-d向量
    """

    # [b, h, w, c] => [b, 1, 1, 1]
    sum_y_true = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true /= eps + sum_y_true

    # [b, h, w, c] => [b, 1, 1, 1]
    sum_y_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred /= eps + sum_y_pred

    # [b, h, w, c]，y_pred为0项直接置0
    loss = tf.where(y_pred != 0, y_true * tf.math.log(y_true / (y_pred + eps) + eps), 0)

    # [b, h, w, c] => # [b, 1, 1, 1] => [1]
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss


def correlation_coefficient(y_true, y_pred, eps=np.finfo('float').eps):
    """
        CC（Correlation Coefficient Loss），是皮尔逊（Pearson）的相关系数，
        将显着性和真实密度图视为测量它们之间线性关系的随机变量。
        首先将每个图像的值除以它们的总和，以产生和为1的分布。

    Args:
        y_true (tensor, float32): 4-d的真实显著性图，值介于0~1之间
        y_pred (tensor, float32): 4-d的预测显著性图，值介于0~1之间

    Returns:
        tensor, float32: 关于误差的0-d向量
       """

    # [b, h, w, c] => [b, 1, 1, 1]
    sum_y_true = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
    sum_y_pred = tf.reduce_sum(y_pred, axis=[1, 2, 3], keepdims=True)

    y_true /= sum_y_true + eps
    y_pred /= sum_y_pred + eps

    mean_x = tf.reduce_mean(y_true, axis=[1, 2, 3])   # [b]
    mean_y = tf.reduce_mean(y_pred, axis=[1, 2, 3])   # [b]

    # mean_x, var_x = tf.nn.moments(y_true, axes=[1, 2, 3])   # [b]
    # mean_y, var_y = tf.nn.moments(y_pred, axes=[1, 2, 3])   # [b]

    # y_true = (y_true - mean_x) / (tf.sqrt(var_x) + eps)
    # y_pred = (y_pred - mean_y) / (tf.sqrt(var_y) + eps)

    mean_prod = tf.reduce_mean(y_true * y_pred, axis=[1, 2, 3])   # [b]
    mean_x_square = tf.reduce_mean(tf.square(y_true), axis=[1, 2, 3])     # E[x^2], [b]
    mean_y_square = tf.reduce_mean(tf.square(y_pred), axis=[1, 2, 3])     # E[y^2], [b]

    num = mean_prod - mean_x * mean_y   # sigma(XY) = E[XY] - E[X]E[Y]

    # sqrt((E[x^2] - E[x]^2)*(E[y^2] - E[y]^2))
    den = tf.sqrt((mean_x_square - tf.square(mean_x)) * (mean_y_square - tf.square(mean_y)))

    return -tf.reduce_mean(num / den)


@tf.function
def nss(y_true, y_pred, eps=np.finfo('float').eps):
    """
        NSS指标（Normalized Scanpath Saliency Loss），
        用于量化眼睛注视位置的显着图值，并用显着图方差对其进行归一化。
        首先将每个图像的值标准化，减去均值再除标准差。

    Args:
        y_true:(tensor, float32): 4-d的二值化显著性图，值为0或1
        y_pred:(tensor, float32): 4-d的预测显著性图，值介于0~1之间

    Returns:
        tensor, float32: 关于误差的0-d向量
    """

    y_mean, y_var = tf.nn.moments(y_pred, [1, 2, 3], keepdims=True)
    y_std = tf.sqrt(y_var)

    y_pred = (y_pred - y_mean) / (y_std + eps)

    # 预测图中对应y_true值为1的均值即为NSS，也可两图点乘求和再求均值，E[XY]/E[X]
    return -tf.reduce_mean((tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) / tf.reduce_sum(y_true, axis=[1, 2, 3])))


if __name__ == '__main__':
    shape = [180, 320]
    y_true = tf.random.uniform([4]+shape+[1], 0, 1, dtype='float32')  # [b, h, w, c]
    y_pred = tf.random.truncated_normal([4]+shape+[1], 0, 1, dtype='float32')  # [b, h, w, c]
    out = nss(y_true, y_pred)

    print('out of loss：', out)
    print('End of program!')
