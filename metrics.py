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


def accuracy_score(y: np.ndarray, y_predict: np.ndarray):
    """
    计算分类问题的准确率
    """
    assert y.shape[0] == y_predict.shape[0], "y和y_predict的大小必须相同"

    return sum(y == y_predict) / len(y)
