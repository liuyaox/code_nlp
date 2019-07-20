#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:49:18 2019
@author: liuyao8
"""

import os
import numpy as np
import pandas as pd
from functools import reduce

import tensorflow as tf
from keras.layers import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras import backend as K


# 设置多GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def get_session(gpu_fraction=1.0):
    """GPU使用设置，何时需要？"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if num_threads:
        tf_config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
    else:
        tf_config=tf.ConfigProto(gpu_options=gpu_options)
    return tf.Session(config=tf_config)


def get_embedding_layer(initializer='uniform', idx2vector=None, vocab_len=10000, embed_dim=64, trainable=False, name=None):
    """
    Create Embedding Layer Using Random Initialization or Pretrained Word Embeddings 
    idx2vector由词汇表word2idx和word2vector生成，保证了idx与embed_matrix能够一致
    """
    assert initializer in ('nonconstant', 'zero', 'uniform', 'normal'), 'initializer must be nonconstant, zero, uniform or normal !'
    if initializer == 'nonconstant':    # 随机初始化Embedding  一般取initializer='uniform'
        embed_layer = Embedding(vocab_len, embed_dim, embeddings_initializer='uniform', trainable=True, name=name)
        
    elif idx2vector is not None:
        # 若提供了idx2vector，则字典大小和Embedding维度大小都由idx2vector决定，否则使用默认值
        vocab_len = len(idx2vector)               # TODO 有时会 +2，为啥子？？？
        embed_dim = len(idx2vector.values()[0])

        # 初始化embed_matrix，当某些idx不存在时，可以有初始化向量  TODO 貌似这种情况不存在？？？
        if initializer == 'zero':
            embed_matrix = np.zeros((vocab_len, embed_dim))                                           # 全零初始化
        elif initializer == 'uniform':
            embed_matrix = np.random.uniform(-0.01, 0.01, (vocab_len, embed_dim))                     # 均匀分布随机初始化
        elif initializer == 'normal':
            vectors = np.stack(idx2vector.values())
            embed_matrix = np.random.normal(vectors.mean(), vectors.std(), (vocab_len, embed_dim))    # 正态分布随机初始化
        
        # TODO idx < vocab_len 之前在哪里见过，作用是啥？？？
        for idx, vector in idx2vector.items():
            if idx < vocab_len:
                embed_matrix[idx, :] = vector
        
        embed_layer = Embedding(vocab_len, embed_dim, embeddings_initializer=Constant(embed_matrix), trainable=trainable, name=name)
        
    else:
        raise ValueError('ERROR, Guy! No idx2vector !')
    return embed_layer


class ParallelModelCheckpoint(ModelCheckpoint):
    """自定义Checkpoint子类，保存模型model的参数(合并多GPU训练得到的参数)"""
    
    def __init__(self, model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)



if __name__ == '__main__':
    
    # 加载模型前设置session，有时候不需要？又好像每次都需要？
    K.set_session(get_session())

    model = xxx
    checkpoint = ParallelModelCheckpoint(
        model, 
        filepath= 'dapei_{epoch:02d}_{val_loss:.4f}.h5', 
        monitor='val_loss', 
        save_best_only=False, 
        save_weights_only=False)
    
    model2 = multi_gpu_model(model, 2)
    model2.compile(xxx)
    history = model2.fit(xxx, callbacks=[checkpoint])
    model.save_weights('xxx.h5')				# 注意是model，而非model2

    print('------------------------ Result -----------------------')
    accs, val_accs = history.history['acc'], history.history['val_acc']
    imax = np.argmax(val_accs)
    print('acc: ' + str(round(accs[imax] * 100, 2)) + '\t' + 'val_acc: ' + str(round(val_accs[imax] * 100, 2)))
    #print('test_acc: ' + str(round(model.evaluate(test_x, test_y)[1] * 100, 2)))


