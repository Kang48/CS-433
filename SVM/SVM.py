import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, batch_size=64):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training with batches
        for _ in range(self.n_iterations):
            # Shuffle data at the beginning of each iteration
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y_ = y_[indices]

            # Process data in batches
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y_[start:end]

                for idx, x_i in enumerate(X_batch):
                    condition = y_batch[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                    if condition:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                    else:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_batch[idx]))
                        self.bias -= self.learning_rate * y_batch[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
