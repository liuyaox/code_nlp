# Word Embedding 降维
>
- version: 1.0
- author: liuyao58


## Description

功能：基于 PCA 的 Word Embedding 降维方法。

脚本说明

- embedding_reduction.py : 可直接当作工具类脚本放入项目中，无其他多余内容

- embedding_reduction.ipynb : 包含 embedding_reduction.py 所有功能，有各种操作记录，过程更加详细

## Evaluation

降维各方法评估效果如下图所示：

![001](../Pics/Word Embedding降维方法评估.jpg)

说明1：Original 表示原始 Word Embedding，Truncated50d 表示按前50位直接截断进行降维，PPA 表示 Post-processing Algorithmm，不进行降维，PCA-nd 表示主成分分析降维为 n 维，其他方法是 PPA 和 PCA 的组合

说明2：评估指标为知识图谱项目中用神经网络模型进行修饰关系二分类的 Accuracy， **只进行了6次重复实验以取均值，并不严谨，仅供参考**

## Reference

- Paper： [Simple & Effective Dimensionality Reduction for Word Embeddings](https://arxiv.org/abs/1708.03629 "Simple & Effective Dimensionality Reduction for Word Embeddings")

- Github：[https://github.com/vyraun/Half-Size](https://github.com/vyraun/Half-Size "https://github.com/vyraun/Half-Size")