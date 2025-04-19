import os
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

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


def categorize_views_svm(views):
    """Categorize views into 2 groups for binary SVM Classification """
    if views < 10000000:  # Less than 10M
        return 0
    else:
        return 1

def svm_classification(X_train, y_train, X_test, y_test):
    
	y_train_svm = np.array([categorize_views_svm(v) for v in y_train])
    
	print("linear")
    # linear SVC
	svm_lin = svm.SVC(kernel='linear', C=1)
	svm_lin.fit(X_train, y_train_svm)
	lin_pred = svm_lin.predict(X_test)
    
	print("RBF")
	# RBF SVC
	svm_rbf = svm.SVC(kernel='rbf', gamma='scale', C=1)
	svm_rbf.fit(X_train, y_train_svm)
	rbf_pred = svm_rbf.predict(X_test)
    
	print("poly")
	# Polynomial SVC
	svm_poly = svm.SVC(kernel='poly', gamma='scale', C=1)
	svm_poly.fit(X_train, y_train_svm)
	poly_pred = svm_poly.predict(X_test)
    
	print("sigmoid")
	# Sigmoid SVC
	svm_sig = svm.SVC(kernel='sigmoid', gamma='scale', C=1)
	svm_sig.fit(X_train, y_train_svm)
	sig_pred = svm_sig.predict(X_test)
     
	check_accuracy(y_test, lin_pred, rbf_pred, poly_pred, sig_pred)


    
def check_accuracy(y_test, lin_pred, rbf_pred, poly_pred, sig_pred):
	y_test_svm = np.array([categorize_views_svm(v) for v in y_test])
	# Print classification report for each SVC
	
	# linear SVC
	lin_accuracy = accuracy_score(y_test_svm, lin_pred)
	print(" Linear SVM acccuracy ", lin_accuracy)
	print(classification_report(y_test_svm, lin_accuracy, labels=[0, 1]))
    
	# RBF SVC
	rbf_accuracy = accuracy_score(y_test_svm, rbf_pred)
	print(" RBF SVM acccuracy ", rbf_accuracy)
	print(classification_report(y_test_svm, rbf_accuracy, labels=[0, 1]))

	# Polynomial SVC
	poly_accuracy = accuracy_score(y_test_svm, poly_pred)
	print(" Polynomial SVM acccuracy ", poly_accuracy)
	print(classification_report(y_test_svm, poly_accuracy, labels=[0, 1]))
     
	# Sigmoid SVC
	sig_accuracy = accuracy_score(y_test_svm, sig_pred)
	print(" Sigmoid SVM acccuracy ", sig_accuracy)
	print(classification_report(y_test_svm, sig_accuracy, labels=[0, 1]))

def stacking_classification(X_train, y_train, X_test, y_test):
    estimators = [
		('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=7, random_state=42)),
        ('ada', AdaBoostClassifier(random_state=42))
    ]
    
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
	
    y_train_binned = np.array([categorize_views(v) for v in y_train])  # Use y_train for binning
    y_test_binned = np.array([categorize_views(v) for v in y_test])  # Use y_test for binning
    clf.fit(X_train, y_train_binned)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test_binned, y_pred, labels=[0, 1]))



def stacking_regressor(X_train, y_train, X_test, y_test):
    estimators = [
		('lr', LinearRegression()),
        ('svr_lin', svm.SVR(kernel='linear')),
        ('ridge', Ridge(random_state=42))
    ]
    print("Stacking Regressor")
    clf = StackingRegressor(estimators=estimators, final_estimator=svm.SVR(kernel='rbf'))
    clf.fit(X_train, y_train)
    
	# Make predictions on test data
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[0, 1]))



# Load preprocessed data
X_train, X_test, y_train, y_test, feature_names = load_data()
svm_classification(X_train, y_train, X_test, y_test)