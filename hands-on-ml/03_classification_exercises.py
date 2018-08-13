## 1. An MNIST Classifier With Over 97% Accuracy

#%%
from enum import Enum

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata

get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt


#%% Download the data

mnist = fetch_mldata('MNIST original')
mnist

#%% Extract the data

X, y = mnist.data, mnist.target

#%% define method to show a digit

def show_digit(digit):
    """Plot a single digit"""
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("on")
    plt.show()

#%% Plot a digit

some_digit = X[36000]
show_digit(some_digit)

#%% Split into a training and test set

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#%% Shuffle the training set so all cross validation folds will be similar

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


#%% Create the KNN classifier

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)

#%% Fitting the simple model

knn_clf.fit(X_train, y_train)

#%% make predictions

y_knn_pred = knn_clf.predict(X_test)

#%% Check the accuracy (0.9714)

knn_score = accuracy_score(y_test, y_knn_pred)

print('knn score on simple model', knn_score)





#%% Tune the hyperparameters with grid search on weights and n_neighbours hyperparameters

param_grid = {'weights': ['uniform', 'distance'], 'n_neighbors': [1, 3]}

knn_clf = KNeighborsClassifier()
knn_grid_clf = GridSearchCV(knn_clf, param_grid, cv=2, n_jobs=2)

#%% Fitting the grid
knn_grid_clf.fit(X_train, y_train)

#%% Examine grid search results

def show_grid_search_results(grid_cv: GridSearchCV):
    """ Print out useful properties of grid search
    """
    print('Best estimator:', grid_cv.best_estimator_)
    print('Best score:', grid_cv.best_score_)
    print('Best params:', grid_cv.best_params_)
    print('CV splits:', grid_cv.n_splits_)

#%% 
pd.DataFrame(knn_grid_clf.cv_results_).sort_values(by='rank_test_score', ascending=False)

#%%
show_grid_search_results(knn_grid_clf)

#%% get the best model for running on the test set
final_model = knn_grid_clf.best_estimator_

# Best estimator: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=3, p=2,
#            weights='distance')
# Best score: 0.967566666667
# Best params: {'n_neighbors': 3, 'weights': 'distance'}
# CV splits: 2

#%% Make predictions over the test set with the best model
final_pred = final_model.predict(X_test)

#%% Calculate the score for the test set
final_score = accuracy_score(y_test, final_pred)

# 0.9717
print('knn score on best model', final_score)


## 2. Data augmentation / training set expansion
# Shift an MNIST image in any direction by one pixel
# Create four shifted copies of each image

#%%
class Direction(Enum):
    """The directions to shift an digit"""
    up = 1
    down = 2
    left = 3
    right = 4

def shift_digit(digit, direction: Direction) -> np.ndarray:
    """Shift a digit in any direction by one pixel."""

    shaped = digit.reshape(28, 28)

    # pad with a zero on all sides
    zero_padded = np.pad(shaped, 1, 'constant', constant_values=0)

    if direction == Direction.up:
        return zero_padded[2:, 1:-1].ravel()
    elif direction == Direction.down:
        return zero_padded[:-2:, 1:-1].ravel()
    elif direction == Direction.left:
        return zero_padded[1:-1, 2:].ravel()
    elif direction == Direction.right:
        return zero_padded[1:-1:, :-2].ravel()
    else:
        # unrecognised direction, return the untransformed input
        return digit

#%% show a digit and the same digit shifted right one pixel
show_digit(some_digit)
shifted_digit = shift_digit(some_digit, Direction.right)
show_digit(shifted_digit)


#%% shift digits
X_augmented = [shift_digit(digit, direction) for digit in X_train for direction in Direction]

y_augmented = [digit for digit in y_train for direction in Direction]

#%% examine the augmented data
len(X_augmented), len(y_augmented)

#%% extend X train

X_train_extended = np.vstack((X_train, X_augmented))

shuffle_index = np.random.permutation(300000)
X_train = X_train_extended[shuffle_index]

print('data augmented', len(X_train_extended), len(X_train))


#%% now do y
y_train_extended = np.append(y_train, y_augmented)

y_train = y_train_extended[shuffle_index]

print('data augmented', len(y_train_extended), len(y_train))


#%% Run extended data set over best model

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=3)
# or knn_clf = KNeighborsClassifier(**knn_grid_clf.best_params_)

#%% Fitting the simple model

knn_clf.fit(X_train, y_train)

#%% make predictions

y_knn_pred = knn_clf.predict(X_test)

#%% Check the accuracy (0.9763)

knn_score = accuracy_score(y_test, y_knn_pred)

print('knn score on simple model', knn_score)



