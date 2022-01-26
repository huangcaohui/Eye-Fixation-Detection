# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers


class ConvLSTMLayer(layers.Layer):
    def __init__(self, shape, kernel, nb_filters_out, nb_filters_att,
                 init=initializers.TruncatedNormal(stddev=0.05),
                 inner_init=initializers.Orthogonal(), attentive_init=initializers.Zeros(),
                 inner_activation=tf.nn.sigmoid, activation=tf.nn.tanh):
        super(ConvLSTMLayer, self).__init__()

        self._kernel = kernel   # 卷积核大小
        self._nb_filters_out = nb_filters_out   # 用于LSTM内部权重卷积的卷积核个数
        self._nb_filters_att = nb_filters_att   # 用于最终二维显著性图(2-d map)输出的卷积核个数
        self._init = init       # 所有权重矩阵（卷积核）和Ua矩阵（卷积核）的初始化值，从均值为0，标准差为0.05的正态分布采样
        self._inner_init = inner_init   # Ui,Uf,Uo,Uc矩阵的初始化值，为随机正交矩阵
        self._attentive_init = attentive_init   # 矩阵Va和所有偏置参数初始化值，为0
        self._inner_activation = inner_activation   # LSTM的归一化函数，为sigma函数
        self._activation = activation   # 图像与显著性图输出激活函数，为tanh函数
        self._size = tf.convert_to_tensor(shape + [self._nb_filters_out])   # 输出的shape

        self.W_a = layers.Conv2D(self._nb_filters_att, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.U_a = layers.Conv2D(self._nb_filters_att, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.V_a = layers.Conv2D(1, self._kernel, padding='same', kernel_initializer=self._attentive_init, use_bias=False)
        self.W_i = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.U_i = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._inner_init, use_bias=True)
        self.W_f = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.U_f = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._inner_init, use_bias=True)
        self.W_o = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.U_o = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._inner_init, use_bias=True)
        self.W_c = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._init, use_bias=True)
        self.U_c = layers.Conv2D(self._nb_filters_out, self._kernel, padding='same', kernel_initializer=self._inner_init, use_bias=True)

    @property
    def state_size(self):
        """
            自定义state_size
        """
        return [self._size, self._size]

    @property
    def output_size(self):
        """
            自定义output_size
        """
        return self._size

    def get_initial_states(self, inputs):
        """
            获取LSTM网络的初始状态值，通过原图像的0核卷积实现
            初始状态包括神经元初始状态cell_state和隐含层状态hidden_state两部分
        """
        # [b, t, h, w, c] => [b, h, w, c]
        initial_state = tf.reduce_sum(inputs, axis=1)    # 在所有时间戳上求和
        # [b, h, w, c] => [b, h, w, _nb_filters_out]
        initial_state = layers.Conv2D(self._nb_filters_out, kernel_size=(1, 1), strides=1, padding='same',
                                      kernel_initializer=self._attentive_init)(initial_state)   # 初始状态全置0
        initial_states = [initial_state, initial_state]     # 初始状态值，包括[cell_state, hidden_state]两部分

        return initial_states

    def call(self, inputs, states):
        """
            update LSTM equations：
            I_t = sigmoid(W_i*X~t+U_i*H_(t-1)+b_i)
            F_t = sigmoid(W_f*X~t~+U_f*H_(t-1)+b_f)
            O_t = sigmoid(W_o*X~t~+U_o*H_(t-1)+b_o)
            G_t = tanh(W_c*X~t+U_c*H_(t-1)+b_c)
            C_t = F_t⊙C_(t-1)+I_t⊙G_t       3-d tensors
            H_t = O_t⊙tanh(C_t)             3-d tensors
            Z_t = v_a*tanh(W_a*X+U_a*H_(t-1)+b_a)
            A_t = softmax(Z_t)
            X~t = A_t⊙X
        """
        x = inputs
        x_shape = x.shape
        h_pre = states[0]   # [b, h, w, _nb_filters_out]
        c_pre = states[1]   # [b, h, w, _nb_filters_out]

        # tanh函数标准化，active_Z = tanh(W_a*X+U_a*H_(t-1))，[b, h, w, c] => [b, h, w, _nb_filters_att]
        active_z = self._activation(self.W_a(x) + self.U_a(h_pre))
        # 与单通道卷积核卷积，Z = V_a*active_Z+b_a，[b, h, w, _nb_filters_att] => [b, h, w, 1]
        z = self.V_a(active_z)

        # 该时刻归一化二维显著性图A，[b, h, w, 1] => [b, h*w] => softmax([b, h*w]) => [b, h, w, 1]
        A = tf.reshape(tf.nn.softmax(layers.Flatten()(z)), x_shape[0:3]+[1])
        # 对特征图在c维度(输入图像通道维度)上平铺，并与原图像点乘获取原图像叠加的显著性图X~t，
        # [b, h, w, 1] => [b, h, w, c] => [b, h, w, c]
        x_title = x*tf.tile(A, [1, 1, 1, x_shape[-1]])

        # 输入门，I_t = sigmoid(W_i*X~t+U_i*H_(t-1)+b_i)，[b, h, w, c] => [b, h, w, _nb_filters_out]
        i = self._inner_activation(self.W_i(x_title) + self.U_i(h_pre))
        # 遗忘门，F_t = sigmoid(W_f*X~t+U_t*H_(t-1)+b_f)，[b, h, w, c] => [b, h, w, _nb_filters_out]
        f = self._inner_activation(self.W_f(x_title) + self.U_f(h_pre))
        # 输出门，O_t = sigmoid(W_o*X~t+U_o*H_(t-1)+b_o)，[b, h, w, c] => [b, h, w, _nb_filters_out]
        o = self._inner_activation(self.W_o(x_title) + self.U_o(h_pre))
        # 输入门额外项，G_t = tanh(W_c*X~t+U_c*H_(t-1)+b_o)，[b, h, w, c] => [b, h, w, _nb_filters_out]
        g = self._activation(self.W_c(x_title) + self.U_c(h_pre))

        # 当前时刻单元状态，C_t = F_t⊙C_(t-1)+I_t⊙G_t，[b, h, w, _nb_filters_out]
        c = f*c_pre + i*g

        # 当前时刻LSTM输出，[b, h, w, _nb_filters_out]
        h = o*self._activation(c)

        return h, tuple([h, c])


class AttentiveConvLSTM(keras.Model):
    def __init__(self, shape, kernel, nb_filters_out, nb_filters_att):
        super(AttentiveConvLSTM, self).__init__()

        self.convlstm = ConvLSTMLayer(shape, kernel, nb_filters_out, nb_filters_att)

    def call(self, inputs, training=None):
        outputs = layers.RNN(self.convlstm, stateful=True)(inputs)

        return outputs


if __name__ == '__main__':
    shape = [180, 320]
    covlstm = AttentiveConvLSTM(shape, 3, 16, 32)
    inputs = tf.random.truncated_normal([4, 4]+shape+[3], dtype='float32')  # [b, t, h, w, c]
    outputs = covlstm(inputs)   # 只输入inputs，默认初始状态全为0
    covlstm.summary()
    print('output shape：', outputs.shape)

    print('End of program!')
