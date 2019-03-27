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



def get_emb_layer(initializer='constant', vocabulary=None, word2vec=None, word2vec_backup=None, emb_dim=64, vocab_len=None, name=None):
    """ Create Embedding Layer Using Random Initialization or Pretrained Word Embeddings """
    if vocab_len is None:
        vocab_len = len(vocabulary)
    # Using random initialization
    if initializer != 'constant':
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=initializer, name=name, trainable=True) # 一般取uniform
    # Using pre-trained word embeddings
    elif word2vec is not None and vocabulary is not None:
        emb_matrix = np.zeros((vocab_len, emb_dim))
        for word, index in vocabulary.items():
            if index < vocab_len:
                if word in word2vec:                    # 首选当前word2vec
                    vector = word2vec.get(word)
                elif word in word2vec_backup:           # 若当前word2vec不存在该word，则使用备用word2vec
                    vector = word2vec_backup.get(word)[:emb_dim]
                else:                                   # 若备用word2vec也不存在该word，则取所有character vector的均值
                    vectors = [word2vec_backup.get(x, np.zeros(emb_dim))[:emb_dim] for x in list(word)]
                    vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
                if vector is not None:
                    emb_matrix[index, :] = vector
        emb_layer = Embedding(vocab_len, emb_dim, embeddings_initializer=Constant(emb_matrix), name=name, trainable=False)
    else:
        print('ERROR! No vocabulary or word2vec or word2vec_backup!')
    return emb_layer



if __name__ == '__main__':
    
    pass