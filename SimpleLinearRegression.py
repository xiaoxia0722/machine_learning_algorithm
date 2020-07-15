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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        根据X_train和y_train训练模型
        """
        assert X_train.ndim == 1, "X_train必须是1维的"
        assert len(X_train) == len(y_train), "X_train和y_train的大小相同"

        X_mean = np.mean(X_train, axis=0)
        y_mean = np.mean(y_train, axis=0)

        num = (X_train - X_mean).dot(y_train - y_mean)
        d = (X_train-X_mean).dot(X_train-X_mean)

        self.a_ = num/d
        self.b_ = y_mean - self.a_*X_mean

        return self

    def predict(self, X_predict: np.ndarray):
        """
        对X_predict进行预测
        """
        assert X_predict.ndim == 1, "X_train必须是1维"
        assert self.a_ is not None and self.b_ is not None, 'predict前需要先进行fit'

        y_predict = X_predict * self.a_ + self.b_
        return y_predict

    def __repr__(self):
        return "SimpleLinearRegression()"
