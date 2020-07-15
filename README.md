# 机器学习算法笔记
主要是一些学习过程中编写的常用的机器学习算法  

使用的语言: Python3  

相关版本说明:
- Python版本:  3.6.8  
- numpy版本: 1.18.1

包含的算法:
- KNN.py: KNN算法(K-近邻算法)
    - KNN函数
        - 参数说明:
            - unknown_point: list, 要预测的数据
            - data_set: numpy.ndarray, 训练集
            - labels: list, 训练集的结果集
            - k: int, 进行判断的最近的点的数量
    - KNNClassifier类:
        - 参数说明:
            - k: int, 进行判断的最近的点的数量
        - 类方法说明: 
            - fit: 训练模型
                - X_train: 训练数据集
                - y_train: 训练数据标签
                - return: self
            - predict: 对数据进行预测
                - X_predict: 预测数据集
                - return: 预测结果
- plotTree.py: 决策树算法(离散值),使用预剪枝
    - 参数说明:
        - data_set: list, 训练集(最后一列为结果集)
        - labels: list, 训练集的特征名
        - value: float, 预剪枝的阈值(使用的是信息增益进行剪枝)

- train_test_split.py: 将数据集分割为训练数据和测试数据
    - train_test_split方法
        - X: 数据集
        - y: 数据集标签
        - test_ratio: 测试数据集比例
        - seed: 随机系数
- metrics.py: 模型评估(预测结果评估)
    - accuracy_score: 分类准确率
    - mean_square_error: 均方误差(MSE)
    - root_mean_square_error: 均方根误差(RMSE)
    - mean_absolute_error: 平均绝对误差(MAE)
    - r2_score: R方值
- preprocessing.py: 数据预处理
    - StandardScaler: 均值方差归一化
        - fit: 训练模型
            - X: 训练数据集
            - return: self
        - transform: 对数据进行归一化处理
            - X: 要进行归一化的数据
            - return: 归一化后的数据集
    - MinMaxScaler: 最值归一化
        - fit: 训练模型
            - X: 训练数据集
            - return: self
        - transform: 对数据集进行归一化
            - X: 需要进行归一化的数据集
            - return: 归一化后的数据集
- SimpleLinearRegression.py: 简单线性回归(单元线性回归)
    - fit: 训练模型
        - X_train: 训练数据集(1维)
        - y_train: 训练数据标签
        - return: self
    - predict: 对数据进行预测
        - X_predict: 需要进行预测的数据集(1维)
- LinearRegression.py: 多元线性回归
    - fit_normal: 训练模型
        - X_train: 训练数据集
        - y_train: 训练数据集对应的标签
        - return: self
    - predict: 对数据进行预测
        - X_predict: 需要进行预测的数据集
        - return: 预测结果集


