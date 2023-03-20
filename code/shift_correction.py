import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, fisher_exact

def analyze_val_data(val_labels, val_preds, test_preds):
    conf_matrix = confusion_matrix(val_labels, val_preds)
    classes = np.unique(val_labels)

    # Compute and report per-class weights (Lipton et al., 2018)
    weights = calc_class_weights(conf_matrix, test_preds, classes)

    return weights

def update_probs(classes, weights, test_preds, test_probs):
    # Update the posteriors
    new_test_probs = adjust_probs(weights, test_probs)

    # Generate new predictions (argmax posterior)
    new_test_preds = np.array([classes[i] for i in
                               np.argmax(new_test_probs, axis=1)])

    return (new_test_preds, new_test_probs)

def normalize_distribution(weights):
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        print('Warning: cannot normalize zero-sum distribution.')
        new_weights = [0.0] * len(weights)
    else:
        new_weights = [w/weight_sum for w in weights]

    return new_weights

def calc_class_weights(conf_matrix, test_preds, classes):
    # Normalize the confusion matrix
    # and transpose it because Lipton et al. assum rows = preds, cols = true
    C_hat = conf_matrix.T*1.0 / np.sum(conf_matrix)

    # Compute the (normalized) class distribution of test set predictions
    n_classes = len(classes)
    n_preds   = len(test_preds)
    mu_hat_test = np.zeros((n_classes,))
    for (i, c) in enumerate(classes):
        mu_hat_test[i] = len([t for t in test_preds \
                              if t == c]) * 1.0 / n_preds

    # Invert the confusion matrix and multiply by mu_hat_test
    try:
        C_hat_inv = linalg.inv(C_hat)
    except:
        print('Failed to invert validation confusion matrix; some classes may be too infrequent.')
        return np.squeeze(np.zeros((n_classes, 1)))
    
    weights = C_hat_inv.dot(mu_hat_test)

    # Clip any negative weights to zero
    weights[weights < 0] = 0.0

    return np.squeeze(weights)

def adjust_probs(val_weights, test_probs):

    new_test_probs = np.zeros_like(test_probs)

    (nitems, nclasses) = test_probs.shape
    # Section 2.2 in [SK20]
    for i in range(nitems):
        for c in range(nclasses):
            new_test_probs[i, c] = val_weights[c] * test_probs[i, c]
        # Normalize new test distribution
        # Override with class weights alone if they canceled out
        if np.sum(new_test_probs[i]) == 0:
            new_test_probs[i] = val_weights
        else:
            new_test_probs[i] = normalize_distribution(new_test_probs[i])

    return new_test_probs                     

