import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using stochastic gradient descent and compute mean squared error.

    Args:
        y: shape=(N, ). The vector of target values.
        tx: shape=(N, D). The matrix of input features.
        initial_w: shape=(D, ). The initial vector of model parameters.
        max_iters: int. The maximum number of iterations to perform.
        gamma: float. The learning rate.

    Returns:
        w: shape=(D, ). The final vector of model parameters.
    """

    def compute_mse(y, tx, w):
        """Compute mean squared error"""
        e = y - tx.dot(w)
        mse = (1 / (2 * len(y))) * np.dot(e, e)
        return mse

    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient
        gradient = -tx.T.dot(y - tx.dot(w)) / len(y)
        # Update weights
        w = w - gamma * gradient
        # Compute current mean squared error
        mse = compute_mse(y, tx, w)
        print(f"Iteration {n_iter+1}/{max_iters}, MSE: {mse}")

    return w


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Performs stochastic gradient descent to minimize the mean squared error (MSE).
    Parameters:
    - y (numpy array): Target values.
    - tx (numpy array): Input feature matrix.
    - initial_w (numpy array): Initial weights.
    - max_iters (int): Maximum number of iterations.
    - gamma (float): Learning rate.
    
    Returns:
    - w (numpy array): Final weights after training.
    """
    def compute_mse(y, tx, w):
        e = y - tx.dot(w)
        mse = (1 / (2 * len(y))) * np.dot(e, e)
        return mse

    def compute_stochastic_gradient(y, tx, w):
        e = y - tx.dot(w)
        gradient = -tx.T.dot(e) / len(y)
        return gradient

    w = initial_w
    for n_iter in range(max_iters):
        for i in range(len(y)):
            # Randomly select a sample
            random_index = np.random.randint(len(y))
            y_n = y[random_index : random_index + 1]
            tx_n = tx[random_index : random_index + 1]
            # Compute stochastic gradient
            gradient = compute_stochastic_gradient(y_n, tx_n, w)
            # Update weights
            w = w - gamma * gradient
        # Compute current mean squared error
        mse = compute_mse(y, tx, w)
        print(f"Iteration {n_iter+1}/{max_iters}, MSE: {mse}")

    return w


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


def compute_mae(y, tx, w):
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


def calculate_accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)


def calculate_f1_score(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_loss(y, tx, w):
    pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)  # ensure pred is a column vector
    loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
    return loss


def calculate_gradient(y, tx, w):
    pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)
    if y.shape != pred.shape:
        y = y.reshape(-1, 1)  # transform y to a column vector
    gradient = np.dot(tx.T, (pred - y)) / len(y)
    return gradient


def least_squares(x_train_pre, y_train, x_test_pre, threshold=0.5):
    """
    Perform linear regression using mini-batch gradient descent and make predictions on test data.

    Args:
        x_train_pre (numpy.ndarray): Training data features, shape (N_train, D).
        y_train (numpy.ndarray): Training data labels, shape (N_train, ).
        x_test_pre (numpy.ndarray): Test data features, shape (N_test, D).
        threshold (float): Threshold for converting continuous predictions to binary class labels.

    Returns:
        numpy.ndarray: Predicted labels for the test data, shape (N_test, ).
    """

    X_train_b = np.c_[
        np.ones((x_train_pre.shape[0], 1)), x_train_pre
    ]  # Add x0 = 1 (bias term)

    # Define batch size
    batch_size = 32  # You can adjust the batch size as needed
    n_samples, n_features = X_train_b.shape
    n_batches = n_samples // batch_size + (
        n_samples % batch_size != 0
    )  # Calculate number of batches

    # Initialize weights
    w = np.zeros(n_features)

    # Linear Regression training using mini-batches
    learning_rate = 0.01  # Learning rate for gradient descent

    for epoch in range(100):  # Number of epochs
        for i in range(n_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, n_samples)
            X_batch = X_train_b[start_index:end_index]  # Get the current batch
            y_batch = y_train[start_index:end_index]  # Corresponding labels

            # Calculate predictions
            y_pred = X_batch.dot(w)

            # Calculate gradient
            gradient = -X_batch.T.dot(y_batch - y_pred) / len(y_batch)  # Mean gradient

            # Update weights
            w -= learning_rate * gradient

    # 3. Prediction function
    def predict(X, w, threshold):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        y_pred_continuous = X_b.dot(w)  # Calculate predictions
        return (y_pred_continuous >= threshold).astype(int)  # Apply threshold

    # Use the trained model to predict on test data
    y_test_pred = predict(x_test_pre, w, threshold=threshold)
    return y_test_pred


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    gradient_descent
    """
    w = initial_w
    y = y.reshape(-1, 1)  
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        if i % 100 == 0:
            print(f"current iteration number：{i}, loss：{loss}")
    
    def predict(tx, w):
        y_pred = sigmoid(np.dot(tx, w))
        y_pred = y_pred.reshape(-1, 1)  
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        return y_pred

    return w, loss, predict


class SVM:
    def __init__(
        self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, batch_size=64
    ):
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
                    condition = (
                        y_batch[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                    )
                    if condition:
                        self.weights -= self.learning_rate * (
                            2 * self.lambda_param * self.weights
                        )
                    else:
                        self.weights -= self.learning_rate * (
                            2 * self.lambda_param * self.weights
                            - np.dot(x_i, y_batch[idx])
                        )
                        self.bias -= self.learning_rate * y_batch[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        lambda_: regularization parameter (scalar)

    Returns:
        w: shape=(D,). The optimal weights vector.
    """
    N, D = tx.shape
    lambda_prime = 2 * N * lambda_
    # Compute the closed-form solution of Ridge Regression
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    return w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=64):
    """
    Regularized logistic regression using mini-batch gradient descent.
    """

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def calculate_loss(y, tx, w, lambda_):
        """
        Compute the regularized loss for logistic regression.
        """
        pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)  # Ensure pred is a column vector
        regularization_term = (lambda_ / 2) * np.sum(w**2)
        loss = (
            -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
            + regularization_term
        )
        return loss

    def calculate_gradient(y, tx, w, lambda_):
        """
        Compute the regularized gradient for logistic regression.
        """
        pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)
        if y.shape != pred.shape:
            y = y.reshape(-1, 1)  # Transform y to a column vector
        gradient = (np.dot(tx.T, (pred - y)) / len(y)) + (lambda_ * w).reshape(
            -1, 1
        ) / len(tx)
        return gradient.flatten()  # Ensure gradient has the same shape as w

    def predict(tx, w):
        """
        Predict the class labels using the logistic regression model.
        """
        y_pred = sigmoid(np.dot(tx, w))
        y_pred[y_pred > 0.3] = 1
        y_pred[y_pred <= 0.3] = 0
        return y_pred

    w = initial_w
    n_samples = len(y)

    for i in range(max_iters):
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        tx = tx[indices]
        y = y[indices]

        # Mini-batch gradient descent
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_tx = tx[start:end]
            batch_y = y[start:end]

            # Compute loss and gradient for the mini-batch
            loss = calculate_loss(batch_y, batch_tx, w, lambda_)
            gradient = calculate_gradient(batch_y, batch_tx, w, lambda_)

            # Update weights
            w = w - gamma * gradient

        # Optionally print progress
        if i % 100 == 0:
            print(f"Current iteration number: {i}, loss: {loss}")

    return w, loss
