# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dcn_vgg import DcnVgg
from ConvLSTM import AttentiveConvLSTM
from gaussian_prior import LearningPrior

import tensorflow as tf
import numpy as np
import loss
import datetime

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class SAMNet(keras.Model):
    """
        SAM模型，包含VGG16，ConvLSTM，Gaussian prior和基本功能函数
    """

    def __init__(self, shape):
        super(SAMNet, self).__init__()

        self.dcn_vgg_outs = None
        self.covnlstm_outs = None
        self.prior_outs = None

        self.shape_img = [1080, 1920]   # 原始图像大小
        self.shape_in = shape      # 输入网络图像大小
        self.shape_vgg_out = list(np.ceil(np.divide(self.shape_in, 8)).astype(np.int))       # DCN-VGG输出图像大小
        self.nb_timestep = 4    # ConvLSTM的时间长度
        self.kernel = 3     # ConvLSTM卷积核大小
        self.nb_vgg_out = 512   # VGG最终输出的通道数
        self.nb_filters_out = 512   # 用于LSTM内部权重卷积和高斯先验的卷积核个数
        self.nb_filters_att = 512   # 用于最终二维显著性图(2-d map)输出的卷积核个数
        self.nb_gaussian = 16    # 高斯先验的高斯函数个数
        self.loss_beta = 2     # CC损失的权值
        self.loss_gamma = 10    # KL散度的权重

        self.dcnvgg = DcnVgg(self.nb_vgg_out)
        self.attentionconvlstm = AttentiveConvLSTM(self.shape_vgg_out, self.kernel, self.nb_filters_out, self.nb_filters_att)
        self.priorlearning1 = LearningPrior(self.shape_vgg_out, self.nb_gaussian, self.nb_filters_out)
        self.priorlearning2 = LearningPrior(self.shape_vgg_out, self.nb_gaussian, self.nb_filters_out)

        # self.layer1 = Sequential([layers.Conv2D(self.nb_filters_out, kernel_size=(5, 5), padding='same', dilation_rate=4),
        #                           layers.BatchNormalization(),
        #                           layers.Activation('relu')])
        # self.layer2 = Sequential([layers.Conv2D(self.nb_filters_out, kernel_size=(5, 5), padding='same', dilation_rate=4),
        #                           layers.BatchNormalization(),
        #                           layers.Activation('relu')])
        self.conv = layers.Conv2D(1, kernel_size=(1, 1), strides=1, padding='same',
                                  activation=tf.nn.relu)

    def _dcn_vgg(self, images, training=None):
        """
            采用VGG13算法，经过13层的卷积网络，将原始VGG13的最后三层全连接层更改为卷积层，
            利用扩张卷积网络（dcn）实现下采样，可以提高更大的感受野，同时又能减小卷积核大小，节省计算资源；
            卷积核的扩张系数为2，空洞卷积，K = k+(k-1)*(dila_rate-1)，k为3，实际得到的K为5。

        Args:
            images (tensor, float32)：包含batchsize的4-D的RGB图像
            training: 当前模式阶段
        """
        # [b, h', w', c] => [b, h'/8, w'/8, nb_vgg_out] = [b, h, w, nb_vgg_out]
        outputs = self.dcnvgg(images, training=training)
        self.dcn_vgg_outs = outputs

        return outputs

    def _attention_convlstm(self, inputs, training=None):
        """
            卷积LSTM网络，通过一种有选择地关注图像不同区域的注意力机制，
            使得卷积LSTM能够专注于输入图像的最显着区域，以迭代地细化预测的显着图。

        Args:
            inputs (tensor, float32)：包含batchsize的4-D的DCN_VGG的输出
            training: 当前模式阶段
        """

        # [b, h, w, nb_vgg_out] => [b, h*w*nb_vgg_out] => [b, h*w*nb_vgg_out*nb_timestep]
        x_tile = tf.tile(layers.Flatten()(inputs), [1, self.nb_timestep])
        # [b, h*w*nb_vgg_out*nb_timestep] => [b, nb_timestep, h, w, nb_vgg_out]
        x_tile = tf.reshape(x_tile, [-1, self.nb_timestep, self.shape_vgg_out[0], self.shape_vgg_out[1], self.nb_vgg_out])
        # [b, nb_timestep, h, w, nb_vgg_out] => [b, h, w, nb_filters_out]
        outputs = self.attentionconvlstm(x_tile, training=training)   # 只输入inputs，默认初始状态全为0
        self.covnlstm_outs = outputs

        return outputs

    def _prior_learning(self, inputs, training=None):
        """
            该模型可以学习一组用高斯函数生成的先验图，以解决人眼注视中存在的中心偏差。
            整个学习先验模块被复制两次。

        Args:
            inputs (tensor, float32)：包含特征的4-D的ConvLSTM的输出
            training: 当前模式阶段
        """
        # [b, h, w, nb_filters_out] => [b, h, w, nb_filters_out]
        priors1 = self.priorlearning1(inputs, training=training)

        # [b, h, w, nb_filters_out] => [b, h, w, nb_filters_out]
        priors2 = self.priorlearning2(priors1, training=training)

        # # [b, h, w, nb_filters_out] => [b, h, w, nb_filters_out]
        # priors1 = self.layer1(inputs)

        # # [b, h, w, nb_filters_out] => [b, h, w, nb_filters_out]
        # priors2 = self.layer2(priors1)

        # 最终的卷积层，[b, h, w, nb_filters_out] => [b, h, w, 1]
        outputs = self.conv(priors2)

        # 上采样，将图片恢复至原大小，[b, h, w, 1] => [b, h', w', 1]
        outputs = tf.image.resize(outputs, self.shape_in)
        self.prior_outs = outputs

        return outputs

    def _normalize(self, maps, eps=np.finfo('float').eps):
        """
            将每一个输出的显著性图归一化到0~1之间。

        Args:
            maps (tensor, float32): 4-D的模型输出
            eps (scalar, float, optional): 无限小数
        """

        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keepdims=True)
        maps -= min_per_image

        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keepdims=True)
        maps = tf.divide(maps, eps + max_per_image)

        return maps

    def call(self, images, training=None):
        """
            通过整个网络架构将RGB图像进行处理，并获取预测的显著性图。
        """

        outputs = self._dcn_vgg(images, training=training)
        outputs = self._attention_convlstm(outputs, training=training)
        outputs = self._prior_learning(outputs, training=training)
        outputs = self._normalize(outputs)

        return outputs

    def net_loss(self, ground_truth_maps, predicted_maps):
        """
            定义整个网络的损失函数。

        Args:
            ground_truth_maps (tensor, float32): 4-D的实际显著性图.
            predicted_maps (tensor, float32):  4-D的预测图.

        Returns:
            tensor, float32: 0-D的平均误差
        """

        kld = loss.kld(ground_truth_maps, predicted_maps)
        cc = loss.correlation_coefficient(ground_truth_maps, predicted_maps)
        error = self.loss_beta*cc + self.loss_gamma*kld

        return error, kld, cc

    def save_model(self, saver, path, form):
        """
            保存模型到checkpoint文件

        Args:
            saver (object): 保存模型的对象
            path (str): 模型的路径
            form (str): 保存模型的类型，'latest' or 'best'
        """
        path = path + form + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        saver.save_weights(path + 'SAM_Model.ckpt')
        print('Saved ' + form + ' weights.')

    def save_log(self, path, form):
        """
            保存模型到logs.

        Args:
            path (str): 模型的路径
            form (str): 保存模型的类型，'train' or 'valid'

        Returns:
            object：logs的保存对象
        """

        path = path + form + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = path + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        print('Created ' + form + ' summary.')

        return summary_writer

    def save_image(self, image_list, path, form, epoch=0, step=0):
        """
            保存图片到指定路径下。

        Args:
            image_list (list, float32): 包含4-D tensor的实际显著性图容器
            path (str): 模型的路径
            form (str): 模型的种类，'train' or 'valid' or 'test'
            epoch (int): 训练集次数，默认为0
            step (int): 步数，默认为0
        """

        path = path + form + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        if not image_list:
            print('No image need to be save!')
            return

        image_num = len(image_list)

        size = image_list[0].shape[1:3]      # 图像的大小
        concat_img = None   # 拼接的图像

        for i in range(image_num):
            image = image_list[i]

            if image.shape[-1] > 1 and image.shape[-1] != 3:
                # 卷积成单通道的
                image = layers.Conv2D(1, kernel_size=(1, 1), strides=1, padding='same', activation=tf.nn.relu)(image)
                image = self._normalize(image)      # 归一化
                image = tf.tile(image, [1, 1, 1, 3])        # 平铺成三通道的
            elif image.shape[-1] == 1:
                image = self._normalize(image)      # 归一化
                image = tf.tile(image, [1, 1, 1, 3])        # 平铺成三通道的

            if i == 0:
                concat_img = image
            else:
                if image.shape[1:3] != size:
                    image = tf.image.resize(image, size)    # 大小不一致则缩放
                concat_img = tf.concat([concat_img, image], axis=0)

        concat_img = concat_img.numpy() * 255.
        concat_img = concat_img.astype(np.uint8)

        col = image_list[0].shape[0]      # 拼接图片的列数，即batch_size
        row = image_num     # 拼接图片的行数

        new_img = Image.new('RGB', (size[1]*col, size[0]*row))     # 空白长图

        for i in range(row):
            for j in range(col):
                img = concat_img[col*i+j]
                img = Image.fromarray(img)
                new_img.paste(img, (size[1]*j, size[0]*i))      # 指定图片左上角坐标

        new_img.save(path + 'SAM%d_%d.png' % (epoch, step))


if __name__ == '__main__':

    path = 'F:/Pycharm projects/deep-learning/Eye_Fixation/model/'
    image_path = 'F:/Pycharm projects/deep-learning/Eye_Fixation/images/'
    shape = [216, 384]

    samnet = SAMNet(shape)
    # inputs = tf.random.truncated_normal([3]+shape+[3], dtype='float32')  # [b, h, w, c]
    # outs = samnet(inputs)
    # samnet.build(input_shape=(None, samnet.shape_in[0], samnet.shape_in[1], 3))

    from dataset import Dataset
    dataset = Dataset(shape, 20, 0.9)
    train_filename = 'E:/HCH/机器学习数据集/3-Saliency-TrainSet/'
    test_filename = 'E:/HCH/机器学习数据集/3-Saliency-TestSet/'
    image_path = 'E:/HCH/vscode project/Eye_Fixation/images/'
    train_db, valid_db, test_db = dataset.get_dataset(train_filename, test_filename)
    inputs, y = next(iter(train_db))

    dcn_vgg_outs = samnet._dcn_vgg(inputs)
    covnlstm_outs = samnet._attention_convlstm(dcn_vgg_outs)
    prior_outs = samnet._prior_learning(covnlstm_outs)
    outs = samnet._normalize(prior_outs)

    # samnet.save_image([inputs, y, dcn_vgg_outs, covnlstm_outs, prior_outs, outs], image_path, form='summary')

    outs = samnet(inputs)
    samnet.summary()
    print('shape of out:', outs.shape)

    print('End of program!')
