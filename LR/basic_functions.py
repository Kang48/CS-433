# -*- coding: utf-8 -*-
#basic functions
#compute_mse
#compute_mae
#calculate_accuracy

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





def calculate_accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred)

def calculate_f1_score(y_true, y_pred):
   
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1








