import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression

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
    - train_logits: [N_train, N_source_labels] f_0(x) logits for the source domain training data
        signifying p(y_s | x)
    - train_labels: [N_train] Corresponding target domain labels for the training data, where
        0 <= each number < N_target_labels
    - validation_logits: [N_val, N_source_labels] f_0(x) logits for the source domain validation data
    - validation_labels: [N_val] Corresponding source domain labels for the validation data

    Outputs:
    - p_source_given_target: [N_source_labels, N_target_labels] conditional probabilities p(y_s | y_t)
        for all possible values of y_s and y_t in the form of a matrix
    """
    
    # Convert logits into probabilities
    # Here we are assuming that the deep model logits f_0(x) are already calibrated
    train_probabilities = softmax(train_logits * 0.8840456604957581)
    validation_probabilities = softmax(validation_logits * 0.8840456604957581)

    # We start with learning the neural network `g` to map source domain probabilities
    # (p(y_s | x), which are treated as a feature vector) to target domain labels y_t.
    # We perform hyperparameter tuning to find the best regularization strength for
    # the logistic model.
    best_classifier = None
    best_accuracy = -1
    best_train_accuracy = -1
    for C in [1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        classifier = LogisticRegression(multi_class='multinomial', C=C, fit_intercept=True, max_iter=1000)
        classifier.fit(train_probabilities, train_labels)

        val_predictions = classifier.predict(validation_probabilities)
        val_accuracy = (val_predictions == validation_labels).mean()
        train_accuracy = (classifier.predict(train_probabilities) == train_labels).mean()
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_classifier = classifier
            best_train_accuracy = train_accuracy
    print(f"Best relationship accuracy = {best_accuracy}")
    print(f"Best train accuracy = {best_train_accuracy}")
    del best_accuracy, best_train_accuracy

    # Now we calibrate the logits of `g` and convert to probabilities p(y_t | y_s)
    # scale = calibrate(validation_logits, validation_labels)
    scale = 1
    p_target_given_source = softmax(best_classifier.coef_.T * scale)
        # p_target_given_source.shape = [N_source_labels, N_target_labels]

    all_probabilities = np.concatenate((train_probabilities, validation_probabilities))
        # all_probabilities.shape = [N_train + N_val, N_source_labels]
    p_source = all_probabilities.sum(axis=0, keepdims=True).T / all_probabilities.shape[0]
        # p_source.shape = [N_source_labels, 1]
    p_joint_distribution = (p_target_given_source * p_source).T
        # p_joint_distribution.shape = [N_target_labels, N_source_labels]
    p_source_given_target = p_joint_distribution / p_joint_distribution.sum(axis=1, keepdims=True)
        # p_source_given_target.shape = [N_target_labels, N_source_labels]
    
    return p_source_given_target
    

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
    logits = torch.tensor(logits).detach()
    labels = torch.tensor(labels).detach()
    t = nn.Parameter(torch.ones(1,1, dtype=torch.float32),requires_grad = True)
    optim = torch.optim.LBFGS([t])

    def loss():
        optim.zero_grad()
        cost = nn.CrossEntropyLoss()(logits*t, labels)
        cost.backward()
        return cost

    for i in range(20):
        optim.step(loss)
        # print("Calibrating: ", t.item())
    
    return t.item()
