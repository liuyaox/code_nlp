#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from functools import reduce
from collections import Iterable


def get_word_embedding(word_embedding_file, header=False, seps=('\t', ',')):
    """ Original Full Word Embedding，用于从中选择出词汇表word2index中的word及其vector，或构建Embedding Layer """
    word_embedding = {}
    with open(word_embedding_file, 'r', encoding='utf-8') as fr:
        if header:
            fr.readline()                        # Drop line 1
        for line in fr:
            values = line.strip().split(seps[0])
            if len(values) >= 2:
                word = values[0]
                vector = values[1:] if seps[0] == seps[1] else values[1].split(seps[1])
                word_embedding[word] = np.asarray(vector, dtype='float32')
    return word_embedding


def get_word2index(corpus, level='word', sep=None):
    """
    词汇表：支持character和word-level，以及两者的汇总
    统计corpus中character/word频率并倒序排序获得index，构建词汇字典：<character/word, index> 后续会使用index来表示character/word
    """
    word2num = {}
    for line in corpus:
        if level in ['character', 'char']:
            objs = list(line.strip())
        elif level == 'word':
            objs = line.strip().split(sep)      # 默认每一行是分词后分隔好的结果
        elif level == 'both':
            objs = list(line.strip()) + line.strip().split(sep)
        for obj in objs:
            if obj in word2num:
                word2num[obj] += 1
            else:
                word2num[obj] = 1
    word_sorted = sorted(word2num, key=word2num.get, reverse=True)          # 按character/word频率倒序排列
    word_list = word_sorted if ' ' in word_sorted else [' '] + word_sorted  # 空格是否加入vocab？ 确保下面word2index中的index从0开始
    word2index = {word: ind for (ind, word) in enumerate(word_list)}        # character/word词汇表：排列rank作为character/word的index   
    return word2index


def get_word2vector(word2index=None, word_embedding=None):
    """ 生成词汇表中的word及其vector，基于Original Full Embedding和词汇表word2index的结合 """
    word2vector = {}
    emb_dim = len(word_embedding.get('a'))
    for word in word2index:
        if word in word_embedding:
            vector = word_embedding.get(word)
        else:
            vectors = [word_embedding.get(x, np.zeros(emb_dim)) for x in list(word)]
            vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
        if vector is not None:
            word2vector[word] = vector
    return word2vector


def get_basic4_dict(corpus, level='word', sep=None, word_embedding):
    """ 4个基础字典，用于各种转换 """
    word2index = get_word2index(corpus, level=level, sep=sep)
    index2word = {ind: word for (word, ind) in word2index.items()}
    word2vector = get_word2vector(word2index, word_embedding)
    index2vector = {ind: word2vector.get(word, -1) for (ind, word) in index2word.items()}
    return word2index, index2word, word2vector, index2vector


def dict_to_2arrays(dic, sortby=None):
    """ 把字典的keys和values转化为2个ndarray  sortby: 按key(=0)或value(=1)排序 """
    if sortby is None:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[sortby])
    keys, values = zip(*items)
    return np.asarray(keys), np.asarray(values)


def array_to_dict(index2key, array):
    """ 把array中的vector按其index转化为dict，key为index2key中index对应的key，value为vector """
    return {index2key.get(ind): vector for (ind, vector) in enumerate(array)}


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
    """ 从word2vector中找到与word0相似度大于thresh的其他word，按相似度排序，相似度计算函数可指定 """
    vector0 = word2vector[word0]
    res = []
    for word, vector in word2vector.items():
        sim = sim_func(vector, vector0)
        if word != word0 and sim >= thresh:
            res.append((word, round(sim,4)))
    return sorted(res, key=lambda x: x[1], reverse=True)


def dict_persist(dic, filename, seps=['\t', ',']):
    """ 字典持久化为文件，每行一个<key, val>对 """
    with open(filename, 'w', encoding='utf-8') as fw:
        for (key, val) in dic.items():
            if not isinstance(val, str) and isinstance(val, Iterable):
                val = seps[1].join([str(x) for x in val])
            fw.write(str(key) + seps[0] + val + '\n')



if __name__ == '__main__':
    
    pass

