# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2020/3/21 16:00
# @Author   : XiaoXia
# @Blog     : https://xiaoxiablogs.top
# @File     : plotTree.py
from math import log
import operator


def get_entropy(data_set: list) -> int:
    """
    计算数据集data_set的熵
    """
    # 获取样例数
    data_set_len = len(data_set)
    # 数据集的结果集
    results = {}
    # 遍历每个样例，获取结果集以及每种结果的数量
    for data in data_set:
        result = data[-1]
        results[result] = results.get(result, 0) + 1
    entropy = 0.0
    for key in results.keys():
        e = float(results[key]) / data_set_len
        entropy -= e * log(e, 2)
    return entropy


def divide_data_set(data_set: list, axis: int, value: str) -> list:
    """
    根据指定的特征axis和特征值value将数据集data_set划分出来所需要的数据集
    """
    # 用于存储需要的数据的列表
    return_data = []
    for data in data_set:
        if data[axis] == value:
            # 将去除指定特征后的样例数据添加到返回列表中
            new_data = data[:axis]
            new_data.extend(data[axis + 1:])
            return_data.append(new_data)
    return return_data


def select_best_axis(data_set: list) -> int:
    """
        选择数据集data_set中信息增益最大的特征值划分
    """
    # 获取特征数
    axis_len = len(data_set[0]) - 1
    # 获取数据集data_set的熵
    data_set_entropy = get_entropy(data_set)
    best_axis, max_entropy = -1, 0.0
    for i in range(axis_len):
        # 获取第i个特征的特征值列表
        axis_list = [value[i] for value in data_set]
        # 当前特征的所有取值
        axis_set = set(axis_list)
        # 用于计算信息增益
        axis_entropy = 0.0
        for axis in axis_set:
            new_data_set = divide_data_set(data_set, i, axis)
            entropy = get_entropy(new_data_set)
            e = float(len(new_data_set)) / len(data_set)
            axis_entropy += e * entropy
        axis_entropy = data_set_entropy - axis_entropy
        if axis_entropy > max_entropy:
            max_entropy = axis_entropy
            best_axis = i
    return best_axis


def get_result(result_list: list):
    """
    返回当前结果集中最多的结果(结果集的一维数组)
    """
    result_dict = {}
    for result in result_list:
        result_dict[result] = result_dict.get(result, 0) + 1
    result_sort = sorted(result_dict.items(), operator.itemgetter(1), reverse=True)
    return result_sort[0][0]


def create_tree(data_set: list, labels: list, value: float = 0.2) -> dict:
    """
    以字典形式创建决策树
    """
    # 获取训练数据的结果集
    result_list = [result[-1] for result in data_set]
    # 如果结果都一样，则返回结果
    if len(set(result_list)) == 1:
        return result_list[0]
    # 如果没有特征则返回最多的结果
    if len(data_set[0]) == 1 or get_entropy(data_set) < value:
        return get_result(result_list)
    # 获取最佳的特征
    best_axis = select_best_axis(data_set)
    # 最佳特征的特征名
    best_axis_label = labels[best_axis]
    # 创建初始决策树
    tree = {best_axis_label: {}}
    del labels[best_axis]
    # 获取最佳特征的所有特征值
    axis_result_set = set([result[best_axis] for result in data_set])
    for result in axis_result_set:
        # 将当前节点的子树进行存储
        tree[best_axis_label][result] = create_tree(divide_data_set(data_set, best_axis, result), labels.copy())
    return tree
