# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, shape, batch_size, train_ratio):
        self.shape_out = shape      # 数据预处理图像缩放大小
        self.batch_size = batch_size    # batch size
        self.train_ratio = train_ratio      # 训练集比例

    def get_dataset(self, train_filename, test_filename):
        train_x, train_y = self.readImage(train_filename)   # 获取所有图片路径全名
        test_x, test_y = self.readImage(test_filename)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, train_size=self.train_ratio, random_state=0, shuffle=True)

        print('number of train sample:', len(train_x))
        print('number of valid sample:', len(valid_x))
        print('number of test sample:', len(test_x))

        self.preprocess(train_x[0], train_y[0])

        train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_db = train_db.shuffle(len(train_x)).map(self.preprocess).batch(self.batch_size)

        valid_db = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        valid_db = valid_db.shuffle(len(valid_x)).map(self.preprocess).batch(self.batch_size)

        test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_db = test_db.map(self.preprocess).batch(self.batch_size)

        # try机制，防止内存或者空iter异常
        try:
            train_sample = next(iter(train_db))
        except StopIteration:
            print(StopIteration)
        else:
            print('train sample:', train_sample[0].shape, train_sample[1].shape)

        return train_db, valid_db, test_db

    def readImage(self, filename, x_image=None, y_image=None):
        if x_image is None:
            x_image = []
        if y_image is None:
            y_image = []

        if not os.path.exists(filename):
            print('The file path does not exist!')
            return

        for file in os.listdir(filename):
            file_path = os.path.join(filename, file)

            if os.path.isdir(file_path):    # 判断是否为文件夹
                self.readImage(file_path, x_image, y_image)     # 递归读取文件夹
            elif os.path.isfile(file_path):     # 判断是否为文件
                if file.endswith('.jpg'):
                    if 'Stimuli' in file_path:      # 原图像
                        x_image.append(file_path)
                    elif 'FIXATIONMAPS' in file_path:   # 注视点图
                        y_image.append(file_path)
                    else:
                        continue
                else:
                    continue
            else:
                print('There are no pictures in the folder!')

        return x_image, y_image

    def preprocess(self, x, y):
        x = tf.io.read_file(x)  # 根据路径读取图片
        x = tf.image.decode_jpeg(x, channels=3)     # 图片解码
        x = tf.image.resize(x, self.shape_out)

        y_map = tf.io.read_file(y)  # 根据路径读取图片
        y_map = tf.image.decode_jpeg(y_map, channels=1)     # 图片解码
        y_map = tf.image.resize(y_map, self.shape_out)

        # y_fix = tf.where(y_map == 0, y_map, 1)   # ground truth fixation

        # 转换成张量并归一化
        # x: [0,255] => 0~1
        x = tf.cast(x, dtype=tf.float32) / 255.
        y_map = tf.cast(y_map, dtype=tf.float32) / 255.

        x = tf.convert_to_tensor(x)
        y_map = tf.convert_to_tensor(y_map)

        return x, y_map


if __name__ == '__main__':
    batch_size = 32
    shape = [270, 480]

    train_filename = 'E:/HCH/机器学习数据集/3-Saliency-TrainSet/'
    test_filename = 'E:/HCH/机器学习数据集/3-Saliency-TestSet/'

    dataset = Dataset(shape, batch_size)
    train_db, valid_db, test_db = dataset.get_dataset(train_filename, test_filename)

    print('End of program!')
