# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/14 7:53
# @File    : metrics.py
# @Site    : 
# @Software: PyCharm
"""
用于存放各种模型预测结果与真实结果的评分方法
"""


import numpy as np
from math import sqrt


def accuracy_score(y: np.ndarray, y_predict: np.ndarray):
    """
    计算分类问题的准确率
    """
    assert y.shape[0] == y_predict.shape[0], "y和y_predict的大小必须相同"

    return sum(y == y_predict) / len(y)


def mean_square_error(y: np.ndarray, y_predict: np.ndarray):
    """
    计算均方误差
    """
    assert y.shape[0] == y_predict.shape[0], "y和y_predict的大小必须相同"
    return np.sum((y_predict - y) ** 2)/len(y)


def root_mean_square_error(y, y_predict):
    """
    计算均方根误差
    """
    assert y.shape[0] == y_predict.shape[0], "y和y_predict的大小必须相同"
    return sqrt(np.sum((y_predict - y) ** 2)/len(y))


def mean_absolute_error(y, y_predict):
    """
    计算平均绝对误差
    """
    assert y.shape[0] == y_predict.shape[0], "y和y_predict的大小必须相同"
    return np.sum(np.absolute(y-y_predict))/len(y)
