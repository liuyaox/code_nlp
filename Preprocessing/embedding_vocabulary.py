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

class CorpusGenerator(object):
    """
    使用 gensim 生成 Word2Vec 所需的语料 Generator，由文件直接生成，支持 word-level 和 char-level
    NOTES
        文件每行必须事先完成分词或分字
    """
    def __init__(self, corpus_file, stopwords=[], sep=' '):
        self.corpus_file = corpus_file
        self.stopwords = stopwords
        self.sep = sep

    def __iter__(self):
        for line in open(self.corpus_file):
            # 一个句子由词或字列表表示，形如：['颜色', '很', '漂亮'] 或 ['颜', '色', '很', '漂', '亮']，过滤指定词或字(如停用词等)
            yield [x for x in line.strip().split(self.sep) if x not in self.stopwords]


def train_w2v_model(sentences, size=100, window=5, min_count=5, workers=3, sg=1, iter=5, compute_loss=True):
    """
    基于语料 corpus 训练 Word2Vec 字/词向量
    ARGS
        sentences: iterable of sentence, 其中sentence是分字/分词列表，形如：['颜色', '很', '漂亮'] 或 ['颜', '色', '很', '漂', '亮']
        其他：与Word2Vec函数参数保持一致，sg=1表示使用skip-gram算法
    RETURN
        model: 训练好的Word2Vec模型，包含(idx, word, vector)三者之间的4种映射字典：idx2word, idx2vector, word2idx, word2vector(即model.wv)
    USAGE
        待完善……
    """
    model = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg, iter=iter, compute_loss=compute_loss)
    model.idx2word = {}
    model.word2idx = {}
    model.idx2vector = {}
    for word in model.wv.vocab.keys():
        idx = model.wv.vocab[word].index            # TODO ？？？
        model.idx2word[idx] = word
        model.word2idx[word] = idx
        model.idx2vector[idx] = model.wv[word]
    return model


# -----------------------------------------------------------------
# 2. 使用预训练的 Word Embedding，得到(idx, word, vector)4种映射关系

def get_word_embedding(word_embedding_file, seps=('\t', ','), header=False):
    """
    Pretrained Word Embedding File --> Original Full Word Embedding
    后续用于从中选择出词汇表word2idx中的word及其vector，或构建Embedding Layer
    """
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


# TODO 构建一个class Vocab，把4个字典及其方法封装在一起！方法可以是Static Method

def get_word2idx(corpus, level='word', sep=None, min_freq=None):
    """
    词汇表 Vocabulary：支持 char-level 和 word-level，以及两者的汇总
    统计 corpus 中 char/word 频率并倒序排序获得 idx，构建词汇字典：<char/word, idx>
    注意：
    其实也可不排序，直接随便赋给每个 char/word 一个 idx，只要保证唯一且固定即可
    比如按加入 Vocabulary 顺序依次赋值为1,2,3,...，0另有所用，比如当作 <PAD>、空格或 <UNK> 的 idx
    TODO idx=0 给谁？？怎么给？？ 也有把PAD和UNK赋值给词汇表里最后2个idx的
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
            word2num[obj] = word2num.get(obj, 0) + 1    # 统计每个 obj 的频次
    # 过滤低频字/词
    if min_freq:
        word2num = {word: num for (word, num) in word2num.items() if num >= min_freq}
    
    word_sorted = sorted(word2num, key=word2num.get, reverse=True)          # 按char/word频率倒序排列
    # TODO 待完善，如何确定 idx=0 对应的 term ???
    word_list = word_sorted if ' ' in word_sorted else [' '] + word_sorted  # 空格是否加入vocab？ 确保下面word2idx中的idx从0开始
    word2idx = {word: idx for (idx, word) in enumerate(word_list)}        # char/word词汇表：排列rank作为char/word的idx   
    return word2idx, word2num


def get_word2vector(word2idx=None, word_embedding=None):
    """ 生成词汇表中的 word 及其 vector，基于 Original Full Embedding 和词汇表 word2idx 的结合 """
    word2vector = {}
    emb_dim = len(word_embedding.get('a'))
    for word in word2idx:
        if word in word_embedding:
            vector = word_embedding.get(word)
        else:
            vectors = [word_embedding.get(x, np.zeros(emb_dim)) for x in list(word)]
            vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
        if vector is not None:
            word2vector[word] = vector
    return word2vector


def get_idx2vector(word2idx, word2vector, word_emb_spare=None):
    """
    基于word2idx和word2vector生成idx2vector，以用于后续生成Embedding Layer Weights
    """
    emb_dim = len(word2vector.values()[0])
    idx2vector = {}
    for word, idx in word2idx.items():
        if word in word2vector:             # 首选当前word2vector
            vector = word2vector.get(word)
        elif word_emb_spare is not None and word in word_emb_spare: # 若当前word2vector不存在该word，则使用备用word_embedding
            vector = word_emb_spare.get(word)[:emb_dim]
        else:                               # 若备用word_embedding为None或也不存在该word，则取所有char-level vector截断后的均值
            vectors = [word_emb_spare.get(x, np.random.uniform(-0.01, 0.01, (emb_dim)))[:emb_dim] for x in list(word)]
            vector = reduce(lambda x, y: x + y, vectors) / len(vectors)
        idx2vector[idx] = vector
    return idx2vector


def get_basic4_dict(corpus, level='word', sep=None, word_embedding=None):
    """ 4个基础字典，用于各种转换 """
    word2idx, _ = get_word2idx(corpus, level=level, sep=sep)
    idx2word = {idx: word for (word, idx) in word2idx.items()}
    word2vector = get_word2vector(word2idx, word_embedding)
    idx2vector = {idx: word2vector.get(word, -1) for (idx, word) in idx2word.items()}   # TODO 不严谨，待完善
    return word2idx, idx2word, word2vector, idx2vector


# -------------------------------------------------------------
# 3. 各种小工具

def get_label2idx(labels):
    """
    Label编码，用于Label向量化
    也可以直接使用Keras.utils.to_categorical函数
    """
    return {label: i for i, label in enumerate(set(labels))}


def seq_to_idxs(seq, word2idx, max_token_len, char_flag=False, unk_idx=0):
    """
    向量化编码：基于词汇表word2idx，把seq转化为idx向量，词汇表中不存在的token使用unk_idx进行编码，适用于特征编码和Label编码
    Char-Level时，seq既可以是分字列表，也可以是字符串(无需空格分隔)，如 ['我', '们', '爱', '学', '习'] 或 '我们爱学习'
    Word-Level时，seq既可以是分词列表，也可以是空格分隔的分词字符串，如 ['我们', '爱', '学习'] 或 '我们 爱 学习'
    """
    if isinstance(seq, str) or isinstance(seq, int) or isinstance(seq, float):
	    seq = list(str(seq)) if char_flag else str(seq).split()
    seq_vec = [word2idx.get(token, unk_idx) for token in seq] # seq中的各个token转化为对应的各个idx
    paddings = [0] * (max_token_len - len(seq_vec))         	# 小于向量长度的部分用0来padding
    return paddings + seq_vec


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
