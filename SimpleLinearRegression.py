# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/14 10:26
# @File    : SimpleLinearRegression.py
# @Site    : 
# @Software: PyCharm
"""
线性回归
"""


import numpy as np


class SimpleLinearRegression:
    """
    使用最小二乘法实现简单线性回归
    """
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        根据x_train和y_train训练模型
        """
        assert x_train.ndim == 1, "x_train必须是1维的"
        assert len(x_train) == len(y_train), "x_train和y_train的大小相同"

        x_mean = np.mean(x_train, axis=0)
        y_mean = np.mean(y_train, axis=0)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train-x_mean).dot(x_train-x_mean)

        self.a_ = num/d
        self.b_ = y_mean - self.a_*x_mean

        return self

    def predict(self, x_predict: np.ndarray):
        """
        对x_predict进行预测
        """
        assert x_predict.ndim == 1, "x_train必须是1维"
        assert self.a_ is not None and self.b_ is not None, 'predict前需要先进行fit'

        y_predict = x_predict * self.a_ + self.b_
        return y_predict

    def __repr__(self):
        return "SimpleLinearRegression()"
