#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:49:18 2019
@author: liuyao8
"""

import numpy as np
import pandas as pd
from functools import reduce
from keras.layers import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf
import os


# 设置多GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def get_session(gpu_fraction=1.0):
    """GPU使用设置，何时需要？"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if num_threads:
        config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
    else:
        config=tf.ConfigProto(gpu_options=gpu_options)
    return tf.Session(config=config)


    
def get_embedding_layer(initializer='constant', word2index=None, word2vector=None, word_embedding_spare=None, 
                  embedding_dim=64, vocab_len=None, trainable=False, name=None):
    """
    Create Embedding Layer Using Random Initialization or Pretrained Word Embeddings 
    word2vector也可以是word_embedding，两者等价，因为word2vector来自于word_embedding，但首选轻便的word2vector
    word_embedding_spare是备用word_embedding，不同于上面word_embedding
    """
    
    if vocab_len is None:
        vocab_len = len(word2index)
        
    if initializer != 'constant':    # 随机初始化的Embedding  一般取initializer='uniform'
        embedding_layer = Embedding(vocab_len, embedding_dim, embeddings_initializer=initializer, trainable=True, name=name)
        
    elif word2index is not None and word2vector is not None:
        emb_matrix = np.zeros((vocab_len, embedding_dim))
        
        for word, index in word2index.items():
            if index < vocab_len:
                if word in word2vector:                     # 首选当前word2vector
                    vector = word2vector.get(word)
                elif word in word_embedding_spare:          # 若当前word2vector不存在该word，则使用备用word_embedding
                    vector = word_embedding_spare.get(word)[:embedding_dim]
                else:                                       # 若备用word_embedding也不存在该word，则取所有character vector的均值
                    vectors = [word_embedding_spare.get(x, np.zeros(embedding_dim))[:embedding_dim] for x in list(word)]
                    vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
                if vector is not None:
                    emb_matrix[index, :] = vector
                    
        embedding_layer = Embedding(vocab_len, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=trainable, name=name)
        
    else:
        raise ValueError('ERROR, Guy! No word2index or word2vector or word_embedding_spare !')
    return embedding_layer



class ParallelModelCheckpoint(ModelCheckpoint):
    """自定义Checkpoint子类，保存模型model的参数(合并多GPU训练得到的参数)"""
    def __init__(self, model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)



if __name__ == '__main__':
    
    # 加载模型前设置session，有时候不需要？
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