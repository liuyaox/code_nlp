#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:49:18 2019
@author: liuyao8
"""

import numpy as np

from keras.layers import Embedding, RepeatVector, Concatenate, Dense, Activation, Dot, LSTM, Input, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras import backend as K


class AttentionModel:
    """Attention模型，Attention打分机制是加性模型(concat)
    借鉴吴恩达课程练习5-3-1: score(ht, hs)=concat([ht, hs])(其他方式有双线性模型general和点积模型dot)
    或此Github: <https://github.com/Choco31415/Attention_Network_With_Keras>
    """
    def __init__(self, config):
        self.Tx = config.Tx        # 输入步长
        self.Ty = config.Ty        # 输出步长
        self.dim_a = config.dim_a  # Encoder状态变量a的维度
        self.dim_s = config.dim_s  # Decoder状态变量s和c的维度
        self.vocab_size = config.vocab_size

        # One Attention for Step Ty
        self.repeator = RepeatVector(self.Tx)           # (m, dim_s) --> (m , Tx, dim_s)
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(10, activation='tanh')
        self.densor2 = Dense(1, activation='relu')
        #self.activator = Activation('softmax', axis=1)  # TODO 务必要注意！ axis=1 对不对？？？
        self.activator = Activation(lambda x: K.softmax(x, axis=1)) # TODO 与上一句有啥区别么？
        self.dotor = Dot(axes=1)        # 注意，axis=1

        # model
        self.model = self.create()
    
    def one_step_attention(self, a, s_prev):
        """为每个单步Ty计算Attention Context: 加性模型"""
        s_prev = self.repeator(s_prev)              # (m, dim_s) -> (m, Tx, dim_s)
        concat = self.concatenator([s_prev, a])     # (m, Tx, dim_s) + (m, Tx, 2*dim_a) -> (m, Tx, dim_s+2*dim_a)  加性模型
        e = self.densor1(concat)                    # -> (m, Tx, 10)
        energies = self.densor2(e)                  # -> (m, Tx, 1)
        alphas = self.activator(energies)           # -> (m, Tx, 1)   TODO 貌似不同于一般情况下Softmax的使用
        context = self.dotor([alphas, a])           # (m, Tx, 1) * (m, Tx, 2*dim_a) -> (m, 1, 2*dim_a)  因为axes=1而非-1
        return context

    def create(self):
        """构造Encoder和Decoder，通过Attention组合在一起，创建模型"""
        # Input
        x = Input(shape=(self.Tx, ), name='input')
        s0 = Input(shape=(self.dim_s, ), name='s0')
        c0 = Input(shape=(self.dim_s, ), name='c0')

        # Encoder
        a = Bidirectional(LSTM(self.dim_a, return_sequence=True))(x)

        # Decoder
        # 注意context维度是(m, 1, 2*dim_a) 可知步长那一维是1，当输入decoder_cell时，LSTM能够自动判断只是一个Cell，并非Layer
        decoder_cell = LSTM(self.dim_s, return_state=True)
        decoder_output = Dense(self.vocab_size, activation='softmax')

        outputs = []
        s, c = s0, c0
        for _ in range(self.Ty):
            context = self.one_step_attention(a, s)                 # 先基于Attention机制生成Context
            s, _, c = decoder_cell(context, initial_state=[s, c])   # Context, 前一步状态s和c，共同输入LSTM，得到当前步状态s和c
            out = decoder_output(s)                                 # 当前步状态s经Softmax后变成输出y
            outputs.append(out)
        
        # Model
        model = Model(inputs=[x, s0, c0], outputs=outputs)
        return model

    def train(self, x_train, s0, c0, y_train, lr, model_path=None):
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()
        # s0, c0 = np.zeros((len(x_train), self.dim_s)), np.zeros((len(x_train), self.dim_s)) # 其实s0,c0可以初始化为零向量
        self.model.fit([x_train, s0, c0], y_train, epochs=30, batch_size=100)
        # self.model.save_weights(model_path)

    def predict(self, x, s0, c0, idx2term=None):
        # s0, c0 = np.zeros((len(x_train), self.dim_s)), np.zeros((len(x_train), self.dim_s)) # 其实s0,c0可以初始化为零向量
        pred = self.model.predict([x, s0, c0])
        pred = np.argmax(pred, axis=-1)
        if idx2term is None:
            return pred
        else:
            return [idx2term[int(idx)] for idx in pred]
    
    




if __name__ == '__main__':
    
    pass

