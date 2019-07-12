#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from keras.utils import to_categorical


if __name__ == '__main__':
    
    # 1. LSA转换 = TFIDF转换 + SVD转换
    # In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. 
    # In that context, it is known as latent semantic analysis (LSA).
    # 原始数据
    train_data = [
        '我也是小白，但是竞赛要取得好的名次我觉得比较简单',
        '因为最重要的已经给你定好了(可以多思考学习举办方的出题方向和方式)',
        '也不用考虑落地问题，剩下的都是偏竞赛方面的技术问题',
        '所以竞赛拿到好的名次并不代表这个人多牛。',
        '但是只要态度端正，不眼高手低，付出一定的时间成本',
        '文本分类任务的目标是想办法预测出文本对应的类别，是NLP的基础任务。',
        '因为数据标注成本相对于其他任务低廉很多，因此有大量的标注数据可以训练模型，这是文本分类性能目前相对较好的重要原因。',
        '接下来我依次介绍三个比赛的任务描述，如果您看完这节迷迷糊糊，请把达观杯的任务描述和目标记住就好。',
        '文本进行过脱敏处理，任务目标是判断文本数据属于什么类别，类别总共有19种。'
    ]
    train_seg = [' '.join(jieba.cut(x)) for x in train_data]  # (9, )  先分词/分字
    val_seg = None

    # TFIDF转换
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(train_seg)
    train_tfidf = tfidf_vectorizer.transform(train_seg)     # .toarray()  (9, 80)
    val_tfidf = tfidf_vectorizer.transform(val_seg)

    # SVD转换
    svd = TruncatedSVD(n_components=6, n_iter=7, random_state=2018)
    svd.fit(train_tfidf)
    train_svd = svd.transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)                      # (9, 6)   80维稀疏向量 -> 6维稠密向量


    # 2. 多标签labels转换  color和cloth
    labels = [
        ('blue', 'jeans'),
        ('blue', 'dress'),
        ('red', 'dress'),
        ('red', 'shirt'),
        ('blue', 'shirt'),
        ('black', 'jeans')
    ]
    mlb = MultiLabelBinarizer()
    labels2 = mlb.fit_transform(labels)
    mlb.classes_                            # 输出：array(['black', 'blue', 'dress', 'jeans', 'red', 'shirt'], dtype=object)  color和cloth混合在一起了！
    mlb.transform([('blue', 'dress')])      # 输出：array([[0, 1, 1, 0, 0, 0]])
    # 注意如下组合！虽然这种组合的label不可能出现，但仍会为其进行转换！其实不用担心应用结果中出现2个color或2个cloth，因为训练数据中就不会有这种组合的label
    mlb.transform([('blue', 'red')])        # 输出：array([[0, 1, 0, 0, 1, 0]])
    # 注意！之后使用labels2训练模型时，loss是binary_entropy而非categorical_entropy，因为模型是将每个输出标签视作一个独立伯努利分布，而且需要独立惩罚每个输出节点


    # 3. 多标签labels转换：labels value -> labels index -> label onehot



