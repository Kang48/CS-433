# -*- coding: utf-8 -*-
#basic functions
#compute_mse
#compute_mae

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D) D is the dimension of the feasures.
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - np.dot(tx, w)  # error (residuals)
    mse = 1 / (2 * len(y)) * np.sum(e**2)
    
    return mse

def compute_mae(y,tx,w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D) D is the dimension of the feasures.
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - np.dot(tx, w)  # error (residuals)
    mae = np.mean(np.abs(e))
    return mae




class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 使用示例
if __name__ == "__main__":
    # 生成一些随机数据
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # 创建并训练线性回归模型
    regressor = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X, y)

    # 进行预测
    predictions = regressor.predict(X)
    print("预测结果:", predictions)








