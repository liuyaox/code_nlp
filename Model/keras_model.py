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



if __name__ == '__main__':
    
    pass