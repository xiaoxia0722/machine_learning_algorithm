# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/14 19:42
# @File    : LinearRegression.py
# @Site    : 
# @Software: PyCharm


import numpy as np


class LinearRegression:
    """
    多元线性回归
    """
    def __init__(self):
        """
        初始化线性回归模型
        """
        self.coef_ = None
        self.interception = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """
        根据X_train和y_train训练模型
        """
        assert X_train.shape[0] == y_train[0], 'X_train和y_train的大小必须相同'

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """
        对x_predict进行预测
        """
        assert X_predict.ndim == 2, "x_train必须是2维"
        assert self.coef_ is not None and self.interception is not None and self._theta is not None, 'predict前需要先进行fit'
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot()

    def __repr__(self):
        return "LinearRegression()"
