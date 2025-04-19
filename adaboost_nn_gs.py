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

# X_train_tensor = tf.convert_to_tensor(X_train)
# X_test_tensor = tf.convert_to_tensor(X_test)
# y_trainbin_tensor = tf.convert_to_tensor(y_train_binned)
# y_testbin_tensor = tf.convert_to_tensor(y_test_binned)
# #Do neural network here
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(num_features, activation='relu'),
#     tf.keras.layers.Dense(num_features/2, activation='relu'),
#     tf.keras.layers.Dense(4)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(X_train_tensor, y_trainbin_tensor, epochs=7)

# test_loss, test_acc = model.evaluate(X_test_tensor, y_testbin_tensor, verbose=2)

# print('\nTest accuracy:', test_acc)

#Adaboosting Regressor 
abregr = AdaBoostRegressor(n_estimators=100)
abregr.fit(X_train, y_train)
y_pred = abregr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n--- Ada Boosting Evaluation ---")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")


#Adaboost Classifier
abclas = AdaBoostClassifier(n_estimators=100)
abclas.fit(X_train, y_train_binned)
y_pred_binned = abclas.predict(X_test)

accuracy = accuracy_score(y_test_binned, y_pred_binned)
print("Accuracy: ", accuracy)

print(classification_report(y_test_binned, y_pred_binned, labels=[0, 1, 2, 3]))

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

#Confusion Matrix for AdaBoostClassifier
cm = confusion_matrix(y_test_binned, y_pred_binned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
disp.plot(cmap=plt.cm.Blues) 
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, 'ab_classifier_confusion_matrix.png'))
plt.close()

#Grid Search with kNN Regression
knn = KNeighborsRegressor()

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: Manhattan, 2: Euclidean
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
y_gs_knn_pred = grid_search.predict(X_test)
print("Best parameters with Grid Search for KNN Regressor:", grid_search.best_params_)
rmse = np.sqrt(mean_squared_error(y_test, y_gs_knn_pred))
mae = mean_absolute_error(y_test, y_gs_knn_pred)
r2 = r2_score(y_test, y_gs_knn_pred)
print("\n--- Grid Search CV with K Nearest Neighbors Regressor ---")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")

def categorize_views_svm(views):
    """Categorize views into 2 groups for binary SVM Classification """
    if views < 10000000:  # Less than 10M
        return 0
    else:
        return 1


#Grid Search with SVM
parameters = {'C':[1, 10, 100]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
y_train_svm = np.array([categorize_views_svm(v) for v in y_train])
y_test_svm = np.array([categorize_views_svm(v) for v in y_test])
clf.fit(X_train, y_train_svm)
y_pred_gs_svm = clf.predict(X_test)
print("Best parameters for Grid Search with SVM:", clf.best_params_)

accuracy = accuracy_score(y_test_svm, y_pred_gs_svm)
print(" GridSearch SVM Accuracy: ", accuracy)

print(classification_report(y_test_svm, y_pred_gs_svm, labels=[0, 1]))

#Confusion Matrix for GridSearchSVM
cm = confusion_matrix(y_test_svm, y_pred_gs_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues) 
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, 'gridsearch_svm_confmatrix.png'))
plt.close()

# Histogram of views after being binned
plt.hist(y_train_binned)
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('View Counts for Each Class')
plt.savefig(os.path.join(results_dir, 'views_binned_histogram.png'))
# Display the plot
plt.close()
