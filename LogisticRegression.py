# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/17 10:50
# @File    : LogisticRegression.py
# @Site    : 
# @Software: PyCharm

import numpy as np


class LogisticRegression:
    """
    逻辑回归
    """

    def __init__(self):
        """
        初始化逻辑回归模型
        """
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta: float = 0.01, n_iters: int = 1e4, n: int = 10):
        """
        根据X_train和y_train训练模型
        """
        assert X_train.shape[0] == len(y_train), 'X_train和y_train的大小必须相同'

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

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
        # initial_theta = np.zeros(X_b.shape[1])
        # self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        #
        # self.intercept_ = self._theta[0]
        # self.coef_ = self._theta[1:]
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

    def predict_classes(self, X_predict):
        """
        对x_predict进行结果预测
        """
        assert X_predict.ndim == 2, "x_train必须是2维"
        assert self.coef_ is not None and self.intercept_ is not None and self._theta is not None, 'predict前需要先进行fit'
        assert X_predict.shape[1] == len(self.coef_)

        proba = self.predict(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def predict(self, X_predict):
        """
        对x_predict进行结果概率预测
        """
        assert X_predict.ndim == 2, "x_train必须是2维"
        assert self.coef_ is not None and self.intercept_ is not None and self._theta is not None, 'predict前需要先进行fit'
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def __repr__(self):
        return "LinearRegression()"

