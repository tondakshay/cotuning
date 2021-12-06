import numpy as np
import torch
import torch.nn as nn

def softmax(z):
    max_els = np.max(z, axis=1, keepdims=True)
    z = z - max_els
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)

def relationship_learning(train_logits, train_labels, validation_logits, validation_labels):
    """
    Returns conditional probabilities of source labels given target labels,
    i.e. p(Y_source = y_s | Y_target = y_t).

    Inputs: 
    - train_logits: [N_train, N_target_labels] f_0(x) logits for the target domain training data
    - train_labels: [N_train] Corresponding target domain labels for the training data
    - validation_logits: [N_train, N_target_labels] f_0(x) logits for the source domain validation data
    - validation_labels: [N_train] Corresponding source domain labels for the validation data

    Outputs:
    - p_source_given_target: [N_source_labels, N_target_labels] conditional probabilities p(y_s | y_t)
        for all possible values of y_s and y_t in the form of a matrix
    """
    pass

def calibrate(logits, labels):
    """
    Returns optimal logit scaling parameter t_best (denoted as t* in the paper),
    where cross entropy loss for scaled logits wrt labels is minimized.

    Inputs:
    - logits: [N] Vector of logits to calibrate
    - labels: [N] Vector of corresponding labels

    Outputs:
    - t_best: [float] t for which cross_entropy loss for logits scaled down by t
        wrt to the labels is minimized
    """
    pass