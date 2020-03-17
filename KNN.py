# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     :  14:42
# @Author   : XiaoXia
# @Blog     : https://xiaoxiablogs.top
# @File     : KNN.py
import numpy as np


def KNN(unknown_point: list, data_set: np.ndarray, labels: list, k: int) -> str:
    """
        将点unknown_point使用KNN(K-最近邻)算法,根据data_set数据进行分类,以最近的k个点为准
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
    # 定义一个字典用于存储前k个点的类别数量
    labels_number = {}

    # 遍历前n个点并且记录他们的类别
    for i in range(k):
        i_label = labels[sort_distance[i]]
        # 将该点对应的类别在字典中增加1，不存在则设置为0并且增加1
        # dict.get()方法用于获取值，第一个参数为要获取的键，第二个参数为字典中没有则要设置成的值
        labels_number[i_label] = labels_number.get(i_label, 0) + 1
    # 对字典中的键值进行降序排序，并且返回(键,值)对列表
    sort_labels_list = sorted(labels_number.items(), key=lambda a: a[1], reverse=True)
    # 将排名最多的类别返回
    return sort_labels_list[0][0]




