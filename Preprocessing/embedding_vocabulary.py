#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from functools import reduce
from collections import Iterable
from gensim.models import Word2Vec


# ---------------------------------------------------------------------
# 1. 使用 gensim 自己训练 Word2Vec，并得到(idx, word, vector)4种映射关系

# TODO 构建一个class Vocab，把4个字典及其方法封装在一起！方法可以是Static Method

# -----------------------------------------------------------------
# 2. 从文件中获得预训练的Word Embedding，并结合word2idx，计算word2vector和idx2vector


# TODO 把Embedding与Vocabulary分开为2个文件！


def get_word2vector_idx2vector(word2idx=None, word_embedding=None, word_emb_spare=None):
    """ 生成词汇表中的 word, idx 及其 vector，基于 Original Full Embedding (或自己训练的Word Embedding)和词汇表 word2idx 的结合 """
    word2vector, idx2vector = {}, {}
    emb_dim = len(word_embedding.get('a'))
    for word, idx in word2idx.items():
        if word in word_embedding:             # 首选当前word embedding
            vector = word_embedding.get(word)
        elif word_emb_spare is not None and word in word_emb_spare: # 若当前word embedding不存在该word，则使用备用word embedding
            vector = word_emb_spare.get(word)[:emb_dim]
        else:
            if word_emb_spare is not None:      # 若备用word embedding中不存在该word, 则取所有char-level vector截断后的均值
                vectors = [word_emb_spare.get(x, np.random.uniform(-0.01, 0.01, (emb_dim)))[:emb_dim] for x in list(word)]
            else:
                vectors = [word_embedding.get(x, np.random.uniform(-0.01, 0.01, (emb_dim))) for x in list(word)]
            vector = reduce(lambda x, y: x + y, vectors) / len(vectors)     # TODO OOV时使用对应的若干字符向量的Average
        if vector is not None:
            word2vector[word] = vector
            idx2vector[idx] = vector
    return word2vector, idx2vector


# -------------------------------------------------------------
# 3. 各种小工具

def get_label2idx(labels):
    """
    Label编码，用于Label向量化
    也可以直接使用Keras.utils.to_categorical或sklearn.LabelBinarizer
    """
    return {label: i for i, label in enumerate(set(labels))}


def dict_to_2arrays(dic, sortby=None):
    """ 把字典的 keys 和 values 转化为2个 ndarray  sortby: 按key(=0)或value(=1)排序 """
    if sortby is None:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[sortby])
    keys, values = zip(*items)
    return np.asarray(keys), np.asarray(values)


def array_to_dict(idx2key, array):
    """ 把 array 中的 vector 按其 idx 转化为 dict，key 为 idx2key 中 idx 对应的 key，value 为 vector """
    return {idx2key.get(idx): vector for (idx, vector) in enumerate(array)}


def dict_persist(dic, filename, seps=['\t', ',']):
    """ 字典持久化为文件，每行一个<key, val>对 """
    with open(filename, 'w', encoding='utf-8') as fw:
        for (key, val) in dic.items():
            if not isinstance(val, str) and isinstance(val, Iterable):
                val = seps[1].join([str(x) for x in val])
            fw.write(str(key) + seps[0] + val + '\n')


def similarity_cos(vec1, vec2):
    """ Compute cosine similarity of 2 vectors """
    if not isinstance(vec1, np.ndarray):
        vec1 = np.asarray(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.asarray(vec2)
    vec_sum = np.sum(vec1 * vec2)
    vec_norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return vec_sum / vec_norm


def get_similar_words(word0, word2vector, sim_func=similarity_cos, thresh=0.7):
    """ 从 word2vector 中找到与 word0 相似度大于 thresh 的其他 word，按相似度排序，相似度计算函数可指定 """
    vector0 = word2vector[word0]
    res = []
    for word, vector in word2vector.items():
        sim = sim_func(vector, vector0)
        if word != word0 and sim >= thresh:
            res.append((word, round(sim,4)))
    return sorted(res, key=lambda x: x[1], reverse=True)



if __name__ == '__main__':
    
    pass
