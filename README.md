# 机器学习算法笔记
主要是一些学习过程中编写的常用的机器学习算法  
使用的语言: Python3  

相关版本说明:
- Python版本:  3.6.8  
- numpy版本: 1.18.1

包含的算法:
- KNN: KNN(K-近邻算法)
    - 参数说明:
        - unknown_point: list, 要预测的数据
        - data_set: numpy.ndarray, 训练集
        - labels: list, 训练集的结果集
        - k: int, 进行判断的最近的点的数量
- plotTree: 决策树算法(离散值),使用预剪枝
    - 参数说明:
        - data_set: list, 训练集(最后一列为结果集)
        - labels: list, 训练集的特征名
        - value: float, 预剪枝的阈值(使用的是信息增益进行剪枝)