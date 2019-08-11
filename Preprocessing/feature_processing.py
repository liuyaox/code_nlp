#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 特征处理：特征生成，特征转换，特征选择，


import numpy as np
import pickle
import jieba
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


# 1. TFIDF特征
# 省略，请参考LSA特征中lsa_vectorizer_2steps中的TFIDF转换，data_tfidf即为TFIDF特征


# 2. LSA特征
# LSA转换 = TFIDF转换 + SVD转换
# In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. 
# In that context, it is known as latent semantic analysis (LSA).

# TODO **kawgs 实现
def lsa_vectorizer(data, ngram_range=(1, 1), stopwords=None, max_features=None, n_components=2, n_iter=5, **kawgs):
    """
    基于数据data训练LSA模型，并生成LSA特征   LSA = TFIDF + SVD
    ARGS
        data: iterable of sentence, sentence是空格分隔的分字/分词字符串
            形如 ['小猫咪 爱 吃肉', '我 有 一只 小猫咪', ...]  假设shape为(9, ) (即9个sentence)
        其他：参数及其默认值与 TfidfVectorizer 和 TruncatedSVD 保持一致
    RETURNs
        lsa: 训练好的LSA模型
        feature: LSA模型应用于data得到的LSA特征
    USAGE  
        训练时，data既可以只是train，也可以是train+val+test，应用时分别应用于train/val/test
    """
    tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=stopwords, sublinear_tf=True, max_features=max_features)  # (9, ) -> (9, max_features)
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=2019)                                       #       -> (9, n_components)
    lsa = make_pipeline(tfidf, svd)
    feature = lsa.fit_transform(data)
    return lsa, feature


def lsa_vectorizer_2steps(data, ngram_range=(1, 1), stopwords=None, max_features=None, n_components=2, n_iter=5, **kawgs):
    """功能同 lsa_vectorizer, 可返回TFIDF和SVD转换器，假设 data 维度为(9, )"""
    # TFIDF 转换
    tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=stopwords, sublinear_tf=True, max_features=max_features)
    data_tfidf = tfidf.fit_transform(data)      # .toarray()  (9, max_features)
    # SVD 转换
    svd = TruncatedSVD(n_components=6, n_iter=7, random_state=2018)
    feature = svd.fit_transform(data_tfidf)     # (9, n_components)   max_features 维稀疏向量 -> n_components 维稠密向量
    return tfidf, svd, feature


# 3. LDA特征



# 4. LSI特征


# 5. 基于卡方统计量，进行特征选择

def occurrence_matrix(texts, categories):
    """
    基于texts和category原始数据，计算token与category的共现矩阵
    ARGS
        texts: iterable, 每个元素是一个token列表, token既可以是token也可以是token id
        categories: iterable, 每个元素是一个类别id，与texts各元素一一对应
    RETURN
        tokens: tokens列表
        matrix: 列表，元素与tokens一一对应，相当于token与category共现矩阵，可用于计算两者卡方统计量，从而进行特征选择(token选择)
    NOTES
        注意，要求categories是向量化后的类别id，且要求类别id从0开始依次递增，如0,1,2,3,...
    """
    cates_num = len(set(categories))
    dic = {}
    for text, cate in zip(texts, categories):
        for token in set(text):
            if token not in dic:
                dic[token] = [0] * cates_num
                dic[token][cate] += 1
            else:
                dic[token][cate] += 1
    tokens = list(dic.keys())
    matrix = list(dic.values())
    return matrix, tokens


def chi2_value(matrix, mask=True):
    """
    基于共现矩阵计算卡方统计量
    ARGS
        matrix: 二维array或list，共现矩阵，以word，document和document category为例，行是word，列是category，某行某列取值表示：当前category下含有当前word的document数量
        mask: 当category下含有word的document数量为0时，是否不再计算category与word的卡方统计量
    RETURN
        values: 卡方统计量，等于(AD-BC)^2*N/((A+B)(A+C)(B+D)(C+D))
    """
    A = np.array(matrix, dtype=np.float)        # A: category下含有word的样本数量，注意类型为float，以便于后续各种复杂计算
    word_sum = np.sum(A, 1).reshape((-1, 1))    # 各行对应的样本数，转化为列向量
    type_sum = np.sum(A, 0)                     # 各列对应的样本数
    N = np.sum(type_sum)                        # N: 总样本数量  各行各列总和
    B = word_sum - A                            # B: 非category下含有word的样本数量
    C = type_sum - A                            # C: category下不含有word的样本数量
    D = N - A - B - C                           # D: 非category下不含有word的样本数量
    # 若针对每一列，当前列内比较各行，而确定某列后，N, A+C, B+D都是确定不变的，可省略
    # 若针对每一行，当前行内比较各列，而确定某行后，N, A+B, C+D都是确定不变的，可省略
    values = N * (A * D - B * C) ** 2 / ((A + B) * (A + C) * (B + D) * (C + D))
    if mask:
        masking = np.sign(A)       # 当A=0时，value应该为0
        values = masking * values
    return values, A, B, C, D, N


def feature_select_by_chi2(matrix, features, max_col_num=1000, mode='column', mask=True):
    """
    基于卡方统计量进行特征选择
    ARGS
        matrix,mask同chi2_value
        features: 特征列表，特征顺序务必要与matrix各行/列保持一致！用于特征索引转换为特征
        max_col_num: 每列可选择的特征数量最大值
        model: 特征选择的模式，column=各列分别选择特征然后汇总选择的特征，max=取特征各列卡方值最大值为特征卡方值从而选择特征，avg=取平均值
    RETURN
        cnter: collections.Counter，类似字典，表示选择的特征，及其被多少列选择
        selected: 列表，表示选择的特征
    """
    values, A, _, _, _, _ = chi2_value(matrix, mask)
    # 共有3种模式进行特征选择
    if mode == 'column':
        masking = np.sign(A)
        col_num = np.sum(masking, 0, dtype=np.int64)    # 各列拥有的特征数量，注意dtype为int，否则为float
        selected = []
        for i in range(A.shape[1]):                     # 遍历各列
            indices = np.argsort(values[:, i])          # 按卡方统计量排序各特征，取其排序索引
            k = min(max_col_num, col_num[i])
            topk = [features[i] for i in indices[-k:]]  # 前k个特征
            selected.extend(topk)
        cnter = Counter(selected)
        return cnter
    elif mode == 'avg':
        value = np.mean(values, axis=1)
    elif mode == 'max':
        value = np.max(values, axis=1)
    else:
        raise ValueError('mode must be column, avg or max !')
    indices = np.argsort(value)
    selected = [features[i] for i in indices[-max_col_num:]]
    return selected



if __name__ == '__main__':
    
    # 1. LSA 转换
    train_data = [    # 分词/分字  (9, )
        '我 也是 小白 ， 但是 竞赛 要 取得 好的 名次 我 觉得 比较 简单',
        '因为 最重要 的 已经 给你 定好 了 ( 可以 多 思考 学习 举办方 的 出题 方向 和 方式)',
        '也 不用 考虑 落地 问题 ， 剩下 的 都是 偏 竞赛 方面 的 技术 问题',
        '所以 竞赛 拿到 好的 名次 并不 代表 这个 人 多 牛。',
        '但是 只要 态度 端正 ， 不 眼高手低 ， 付出 一定 的 时间 成本',
        '文本 分类 任务 的 目标 是 想办法 预测 出 文本 对应 的 类别 ， 是 NLP 的 基础 任务。',
        '因为 数据 标注 成本 相对于 其他 任务 低廉 很多 ， 因此 有 大量 的 标注 数据 可以 训练 模型 ， 这是 文本 分类 性能 目前 相对 较好 的 重要 原因。',
        '接下来 我 依次 介绍 三个 比赛 的 任务 描述 ， 如果 您 看完 这节 迷迷糊糊 ， 请 把 达观 杯 的 任务 描述 和 目标 记住 就好。',
        '文本 进行 过 脱敏 处理 ， 任务 目标 是 判断 文本 数据 属于 什么 类别 ， 类别 总共 有 19 种 。'
    ]
    val_data = None
    lsa, train_lsa = lsa_vectorizer(train_data, n_components=6, n_iter=7)
    val_lsa = lsa.transform(val_data)


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


    # 3. 多标签labels转换：labels value -> labels idx -> label onehot


    # 4. 基于卡方统计量进行特征选择
    texts = [['t1', 't2', 't3', 't4'], ['t2', 't3', 't5'], ['t1', 't4', 't5'], ['t2','t4'], ['t3', 't4'], ['t1', 't3', 't4']]
    categories = [1, 2, 0, 1, 0, 1]
    matrix, tokens = occurrence_matrix(texts, categories)
    cnter = feature_select_by_chi2(matrix, tokens)  # cnter即为选择的特征，及其被选择的次数
