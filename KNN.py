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


class KNNClassifier:
    def __init__(self, k: int):
        assert 1 <= k, "k必须是有效的"
        self.k = k
        self._X_train = np.array([])
        self._y_train = np.array([])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.shape[0] == y_train.shape[0], '训练数据和标签数量必须相同'
        assert self.k <= X_train.shape[0], "训练数据集的数量必须大于k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict: np.ndarray):
        assert self._X_train is not None and self._y_train is not None, "必须训练这个模型"
        assert X_predict.shape[1] == self._X_train.shape[1], "预测数据必须与训练数据的特征数相同"

        y_predict = []

        for x in X_predict:
            # 计算未知点与训练数据集的距离
            distance = [sqrt(np.sum((data - x) ** 2)) for data in self._X_train]
            # 对距离进行排序
            distance_argsort = np.argsort(distance)

            # 获取距离最近的前k个数据的标签
            topk_label = [self._y_train[i] for i in distance_argsort[:self.k]]
            # 统计每种标签的数量
            labels_count = Counter(topk_label)
            # 返回数量最多的标签(未知点的分类结果)
            y_predict.append(labels_count.most_common(1)[0][0])
        return np.array(y_predict)

    def __repr__(self):
        return "KNN(k=%d" % self.k
