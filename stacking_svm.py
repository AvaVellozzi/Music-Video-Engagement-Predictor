import os
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
#import tensorflow as tf


# Import the modularized preprocessing
from preprocessing import load_data

def categorize_views(views):
    """Categorize views into 4 groups after removing >1B category"""
    if views < 1000000:  # Less than 1M
        return 0
    elif views < 10000000:  # 1M - 10M
        return 1
    elif views < 100000000:  # 10M - 100M
        return 2
    else:  # 100M - 1B (since we removed >1B)
        return 3

# Load preprocessed data
X_train, X_test, y_train, y_test, feature_names = load_data()
y_train_binned = np.array([categorize_views(v) for v in y_train])  # Use y_train for binning
y_test_binned = np.array([categorize_views(v) for v in y_test])  # Use y_test for binning
num_features = X_train.shape[1]

