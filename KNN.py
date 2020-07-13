# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     :  14:42
# @Author   : XiaoXia
# @Blog     : https://xiaoxiablogs.top
# @File     : KNN.py
import numpy as np
from collections import Counter


def KNN(unknown_point: list, data_set: np.ndarray, labels: list, k: int) -> str:
    """
        将点unknown_point使用KNN(K-近邻算法),根据data_set数据进行分类,以最近的k个点为准
    """
    # 获取已知分类结果的点的数量
    data_set_size = len(data_set)
    # 将unknown_point扩展为data_set_size维,并求得与每个点的差
    unknown_point = np.tile(unknown_point, (data_set_size, 1)) - data_set
    # 将差进行平方，得到x方和y方
    sq_point = unknown_point ** 2
    # 将每个维度进行求和，即求得x和y的平方和
    sq_distances = sq_point.sum(axis=1)
    # 对平方和进行开方，求得距离
    distances = sq_distances ** 0.5
    # 对距离进行排序，得到排序后相应的下标
    sort_distance = distances.argsort()
    # 找到前k个点
    k_number = [sort_distance[i] for i in sort_distance[:k]]
    # 统计前k个点的类别数量
    votes = Counter(k_number)
    # 返回最大的数量的标签
    return votes.most_common(1)[0][0]


