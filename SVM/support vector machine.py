#support vector machine
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # 将标签转换为-1和1

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

# 使用示例
if __name__ == "__main__":
    # 生成一些随机数据
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.random.choice([-1, 1], size=100)

    # 创建并训练SVM模型
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    svm.fit(X, y)

    # 进行预测
    predictions = svm.predict(X)
    print("预测结果:", predictions)