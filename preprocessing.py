# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/14 9:05
# @File    : preprocessing.py
# @Site    : 
# @Software: PyCharm
"""
用于处理数据的各种方法
"""


import numpy as np


class StandardScaler:
    """
    用于数据归一化处理
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        """
        根据X获取数据的均值和方差
        """
        assert X.ndim == 2, "X必须是二维数据"

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X: np.ndarray):
        """
        对X进行均值方差归一化处理
        """
        assert X.ndim == 2, "X必须是二维数据"
        assert self.mean_ is not None and self.scale_ is not None, "必须先进行fit"
        assert X.shape[1] == len(self.mean_), "X的特征数必须和训练数据的特征数相同"

        resX = (X - self.mean_)/self.scale_
        return resX


class MinMaxScaler:
    """
    最值归一化
    """

    def __init__(self):
        self.max_ = None
        self.min_ = None

    def fit(self, X: np.ndarray):
        """
        获取X的最大值和最小值
        """
        assert X.ndim == 2, 'X必须是二维数据'

        X = X.astype('float')
        self.max_ = np.max(X, axis=0)
        self.min_ = np.min(X, axis=0)

        return self

    def transform(self, X: np.ndarray):
        """
        对X进行最值归一化处理
        """
        assert X.ndim == 2, "X必须是二维数据"
        assert self.max_ is not None and self.min_ is not None, "必须先进行fit"
        assert X.shape[1] == len(self.max_), "X的特征数必须和训练数据的特征数相同"

        X = X.astype('float')
        resX = (X-self.min_)/(self.max_ - self.min_)
        return resX
