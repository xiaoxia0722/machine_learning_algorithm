# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     :  14:42
# @Author   : XiaoXia
# @Blog     : https://xiaoxiablogs.top
# @File     : KNN.py
import numpy as np
from collections import Counter
from math import sqrt


def KNN(unknown_point: np.ndarray, data_set: np.ndarray, labels: np.ndarray, k: int) -> str:
    """
    将点unknown_point使用KNN(K-近邻算法),根据data_set数据进行分类,以最近的k个点为准
    """
    # 对传入的参数进行判断
    assert 1 <= k <= data_set.shape[0], "k必须是有效的"
    assert data_set.shape[0] == labels.shape[0], '训练数据和标签数量必须相同'
    assert unknown_point.shape[0] == data_set.shape[1], '待分类数据需要和训练数据特征数相同'

    # 计算未知点与训练数据集的距离
    distance = [sqrt(np.sum((data - unknown_point)**2)) for data in data_set]
    # 对距离进行排序
    distance_argsort = np.argsort(distance)

    # 获取距离最近的前k个数据的标签
    topk_label = [labels[i] for i in distance_argsort[:k]]
    # 统计每种标签的数量
    labels_count = Counter(topk_label)
    # 返回数量最多的标签(未知点的分类结果)
    return labels_count.most_common(1)[0][0]
