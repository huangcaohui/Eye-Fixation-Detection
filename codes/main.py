# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import Dataset
from sam_model import SAMNet

import tensorflow as tf
import numpy as np
import random

from tensorflow import optimizers

random.seed(22)
np.random.seed(22)
tf.random.set_seed(22)

PARAMS = {
    "epochs": 50,
    "batch_size": 20,
    "train_radio": 0.9,         # 划分的训练集比例
    "learning_rate": 2e-4,      # 学习率
    "reg_lambda": 2e-6,     # 正则化系数
    "shape_height": 216,
    "shape_width": 384,
}


def train_model(shape, train_db, valid_db, model_path, log_path, image_path):
    epochs = PARAMS['epochs']
    lr = PARAMS['learning_rate']
    reg_lambda = PARAMS['reg_lambda']
    optimizer = optimizers.Adam(lr)

    samnet = SAMNet(shape)

    summary_writer_train = samnet.save_log(log_path, 'train')
    summary_writer_valid = samnet.save_log(log_path, 'valid')
    valid_history = []

    for epoch in range(epochs):
        # 训练集训练
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                outputs = samnet(x, training=True)
                netloss, kld, cc = samnet.net_loss(y, outputs)  # 损失函数

                loss_regularization = []
                for p in samnet.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

                netloss = netloss + reg_lambda*loss_regularization      # 添加正则化项

            grads = tape.gradient(netloss, samnet.trainable_variables)
            optimizer.apply_gradients(zip(grads, samnet.trainable_variables))

            if step % 40 == 0:
                print('train:', epoch, step, 'loss:', float(netloss), 'kld:', float(kld), 'cc:', float(-cc))
                image = [x, y, samnet.dcn_vgg_outs, samnet.covnlstm_outs, samnet.prior_outs, outputs]
                samnet.save_image(image, image_path, 'train', epoch, step)

        with summary_writer_train.as_default():
            tf.summary.scalar('train-loss', float(netloss), step=epoch)
            tf.summary.scalar('train-KLD', float(kld), step=epoch)
            tf.summary.scalar('train-CC', float(-cc), step=epoch)

        # 验证集验证
        total_num = 0
        total_loss = 0
        total_kld = 0
        total_cc = 0
        for step, (x, y) in enumerate(valid_db):
            outputs = samnet(x, training=False)
            netloss, kld, cc = samnet.net_loss(y, outputs)  # 损失函数
            total_num += 1
            total_loss += float(netloss)
            total_kld += float(kld)
            total_cc += float(-cc)

            if step % 3 == 0:
                image = [x, y, samnet.dcn_vgg_outs, samnet.covnlstm_outs, samnet.prior_outs, outputs]
                samnet.save_image(image, image_path, 'valid', epoch, step)

        avg_loss = float(total_loss/total_num)
        avg_kld = float(total_kld/total_num)
        avg_cc = float(total_cc/total_num)

        print('valid: average loss:', avg_loss, 'average kld', avg_kld, 'average cc:', avg_cc)

        valid_history.append(avg_loss)

        if avg_loss == min(valid_history):
            samnet.save_model(samnet, model_path, 'best')

        with summary_writer_valid.as_default():
            tf.summary.scalar('valid-loss', avg_loss, step=epoch)
            tf.summary.scalar('valid-KLD', avg_kld, step=epoch)
            tf.summary.scalar('valid-CC', avg_cc, step=epoch)

        samnet.save_model(samnet, model_path, 'latest')
    return samnet


def test_model(shape, test_db, model_path, image_path, samnet=None):
    samnet = SAMNet(shape)

    best_model_path = model_path + 'best' + '/'
    if os.path.exists(best_model_path):
        samnet.load_weights(best_model_path + 'SAM_Model.ckpt')  # 如果有模型则直接加载模型参数

    total_num = 0
    total_kld = 0
    total_cc = 0

    for step, (x, y) in enumerate(test_db):
        outputs = samnet(x, training=False)
        _, kld, cc = samnet.net_loss(y, outputs)  # 损失函数
        total_num += 1
        total_kld += float(kld)
        total_cc += float(-cc)

        image = [x, y, samnet.dcn_vgg_outs, samnet.covnlstm_outs, samnet.prior_outs, outputs]
        samnet.save_image(image, image_path, 'test', step=step)
        print('test:', step, 'kld:', float(kld), 'cc:', float(-cc))

    print('test average kld:', float(total_kld/total_num), 'average cc:', float(total_cc/total_num))


if __name__ == '__main__':
    train_filename = 'E:/HCH/机器学习数据集/3-Saliency-TrainSet/'
    test_filename = 'E:/HCH/机器学习数据集/3-Saliency-TestSet/'
    model_path = 'E:/HCH/vscode project/Eye_Fixation/model/'
    log_path = 'E:/HCH/vscode project/Eye_Fixation/logs/'
    image_path = 'E:/HCH/vscode project/Eye_Fixation/images/'

    shape = [PARAMS['shape_height'], PARAMS['shape_width']]
    batchsz = PARAMS['batch_size']

    shape = [PARAMS['shape_height'], PARAMS['shape_width']]
    batchsz = PARAMS['batch_size']
    train_radio = PARAMS['train_radio']

    dataset = Dataset(shape, batchsz, train_radio)
    train_db, valid_db, test_db = dataset.get_dataset(train_filename, test_filename)

    samnet = train_model(shape, train_db, valid_db, model_path, log_path, image_path)
    test_model(shape, test_db, model_path, image_path, samnet)
