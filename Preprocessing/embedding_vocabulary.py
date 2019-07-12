#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from functools import reduce
from collections import Iterable
from gensim.models import Word2Vec


class CorpusGenerator(object):
    """使用 gensim 生成 Word2Vec 所需的语料 Generator，由文件直接生成，要求文件每行事先完成分词或分字，支持 word-level 和 char-level"""
    def __init__(self, corpus_file, filter_set=[], sep=' '):
        self.corpus_file = corpus_file
        self.filter_set = filter_set
        self.sep = sep

    def __iter__(self):
        for line in open(self.corpus_file):
            # 一个句子由词或字列表表示，形如：['颜色', '很', '漂亮'] 或 ['颜', '色', '很', '漂', '亮']，过滤指定词或字(如停用词等)
            yield [x for x in line.strip().split(self.sep) if x not in self.filter_set]



def get_word_embedding(word_embedding_file, seps=('\t', ','), header=False):
    """ Original Full Word Embedding，用于从中选择出词汇表 word2index 中的 word 及其 vector，或构建 Embedding Layer """
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
    词汇表：支持 character 和 word-level，以及两者的汇总
    统计 corpus 中 character/word 频率并倒序排序获得 index，构建词汇字典：<character/word, index> 后续会使用 index 来表示 character/word
    注意：其实也可不排序，直接随便赋给每个 character/word 一个 index，只要保证唯一且固定即可
    比如按加入 Vocabulary 顺序依次赋值为1,2,3,...，0另有所用，比如当作 <PAD>、空格或 <UNK> 的 index
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
    """ 生成词汇表中的 word 及其 vector，基于 Original Full Embedding 和词汇表 word2index 的结合 """
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
    """ 把字典的 keys 和 values 转化为2个 ndarray  sortby: 按key(=0)或value(=1)排序 """
    if sortby is None:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[sortby])
    keys, values = zip(*items)
    return np.asarray(keys), np.asarray(values)


def array_to_dict(index2key, array):
    """ 把 array 中的 vector 按其 index 转化为 dict，key 为 index2key 中 index 对应的 key，value 为 vector """
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
    """ 从 word2vector 中找到与 word0 相似度大于 thresh 的其他 word，按相似度排序，相似度计算函数可指定 """
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


def seq_to_vector(seq, vocabulary, max_token_len, char_flag=False):
    """基于词汇表 vocabulary，把 seq 转化为向量"""
    if isinstance(seq, str) or isinstance(seq, int) or isinstance(seq, float):
		tokens = list(str(seq)) if char_flag else str(seq).split()
    seq_vec = [vocabulary.get(token, 0) for token in tokens] 	# tokens中的各个token转化为对应的各个index
    paddings = [0] * (max_token_len - len(seq_vec))         	# 小于向量长度的部分用0来padding
    return paddings + seq_vec



if __name__ == '__main__':
    
    # 使用 gensim 生成 word2vec
    sentences = CorpusGenerator('xxx.txt', stop_word_set)
    model = Word2Vec(sentences, sg=1, size=100, compute_loss=True, window=5, workers=8, iter=8, min_count=2)

