# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/7/14 7:14
# @File    : train_test_split.py
# @Site    : 
# @Software: PyCharm

import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: float = None):
    """
    将数据X,y按照test_ratio进行分割,分割为X_train, X_test, y_train, y_test
    """
    # 判断数据是否合理
    assert X.shape[0] == y.shape[0], "X和y的大小必须相同"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio必须有效"

    # 如果指定生成固定随机数
    if seed:
        np.random.seed(seed)

    # 生成随机索引
    random_indexes = np.random.permutation(len(X))

    # 生成测试数据和训练数据的索引
    test_size = int(len(X) * test_ratio)
    test_indexes = random_indexes[: test_size]
    train_indexes = random_indexes[test_size:]

    # 生成测试数据和训练数据
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
