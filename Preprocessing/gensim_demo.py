# -*- coding: utf-8 -*-
"""
Created:    Fri Apr 19 16:52:26 2019
Author:     liuyao8
Descritipn: 
"""


from gensim.models import Word2Vec, KeyedVectors, Phrases
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.scripts.glove2word2vec import glove2word2vec


# 0. 通用数据和函数
# common_texts: list of list，每个list表示一个文档或句子的分词结果
# get_tmpfile(fname)：与temporary目录拼接，表示临时目录下的某文件，如C:\\Users\\liuyao8\\AppData\\Local\\Temp\\<fname>
# datapath(fname): os.path.join(module_path, 'test_data', fname)，表示当前模块的测试目录下的某文件

# 1. 使用corpus训练word2vector
# 1.1 初始训练
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
path = get_tmpfile("word2vec.model")
model.save(path)

# 1.2 加载模型并继续训练(流式训练，reading data from disk on-the-fly)
model = Word2Vec.load(path)
model.train([["hello", "world"]], total_examples=1, epochs=1)

# 1.3 训练得到word2vector  
word2vector = model.wv              # KeyedVectors
vector = word2vector['computer']    # numpy vector of shape (100, )

path = get_tmpfile("wordvectors.kv")
word2vector.save(path)
word2vector = KeyedVectors.load(path, mmap='r')

# 1.4 自动检测并训练词组Phrase
bigram_transformer = Phrases(common_texts)
model = Word2Vec(bigram_transformer[common_texts], min_count=1)


# 2 word embedding
# 2.1 加载现成的word embedding
word2vector1 = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)       # C text format
word2vector2 = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)    # C bin format


# 2.2 转化普通glove文件为Gensim支持的word2vec格式，即 C text format
# 普通glove文件格式：没有header，从第一行开始就是word及其vector，空格分隔
# word2vec文件格式：第一行是vector个数和vector维度，其他行同普通txt文件，空格分隔
glove_file = './data/normal_glove.txt'
word2vec_file = './data/normal_word2vec.txt'
glove2word2vec(glove_file, word2vec_file)
word2vector = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)


# Get a Keras 'Embedding' layer with weights set as the Word2Vec model's learned word embeddings.
embedding_layer = word2vector.get_keras_embedding(train_embeddings=False)
