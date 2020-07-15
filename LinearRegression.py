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
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """
        根据X_train和y_train训练模型
        """
        assert X_train.shape[0] == len(y_train), 'X_train和y_train的大小必须相同'

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta: float = 0.01, n_iters: int = 1e4, n: int = 10):
        """
        根据X_train和y_train训练模型
        """
        assert X_train.shape[0] == len(y_train), 'X_train和y_train的大小必须相同'

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        for i in range(n):
            initial_theta = np.random.random(X_b.shape[1])
            theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
            if self._theta is None:
                self._theta = theta
            elif J(theta, X_b, y_train) < J(self._theta, X_b, y_train):
                self._theta = theta

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """
        对x_predict进行预测
        """
        assert X_predict.ndim==2, "x_train必须是2维"
        assert self.coef_ is not None and self.intercept_ is not None and self._theta is not None, 'predict前需要先进行fit'
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def __repr__(self):
        return "LinearRegression()"
