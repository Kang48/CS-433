#logistic_regression

import numpy as np
from basic_functions import *


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_loss(y, tx, w):
    pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)  #ensure pred is a column vector
    loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
    return loss


def calculate_gradient(y, tx, w):
    pred = sigmoid(np.dot(tx, w)).reshape(-1, 1)  
    if y.shape != pred.shape:
        y = y.reshape(-1, 1)  # transform y to a column vector
    gradient = np.dot(tx.T, (pred - y)) / len(y)
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    gradient_descent
    """
    w = initial_w
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        if i % 100 == 0:
            print(f"current iteration number：{i}, loss：{loss}")
    return w, loss

def predict(tx, w):
    y_pred = sigmoid(np.dot(tx, w))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return y_pred