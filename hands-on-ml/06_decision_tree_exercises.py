# 7. Train and fine-tune a Decision Tree for the moons dataset.
#%%   a. Generate a moons dataset using make_moons(n_samples=10000, noise=0.4).
    
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

#%%   b. Split it into a training set and a test set using train_test_split().
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#%%   c. Use grid search with cross-validation (with the help of the GridSearchCV class) to find
#     good hyperparameter values for a DecisionTreeClassifier. Hint: try various values for
#     max_leaf_nodes.
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# simple classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

score = accuracy_score(y_test, y_pred)

score # 0.806


# grid search
param_grid  = {'max_depth': [3, 10, 30], 'max_leaf_nodes': [3, 5, 10, 20, 30]}

tree_grid_clf = GridSearchCV(tree_clf, param_grid)


#%%   d. Train it on the full training set using these hyperparameters, and measure your model’s
#     performance on the test set. You should get roughly 85% to 87% accuracy.

tree_grid_clf.fit(X_train, y_train)

y_pred = tree_grid_clf.predict(X_test)

score = accuracy_score(y_test, y_pred)

print(score) # 0.8648

tree_grid_clf.best_estimator_


# 8. Grow a forest.
#%%   a. Continuing the previous exercise, generate 1,000 subsets of the training set, each containing
#     100 instances selected randomly. Hint: you can use Scikit-Learn’s ShuffleSplit class for
#     this.
import numpy as np

ss = ShuffleSplit(n_splits=1000, random_state=42, test_size=0, train_size=100)

#classifiers = []

predictions = list()
scores = []

best_params = tree_grid_clf.best_estimator_.get_params()

for train_index, test_index in ss.split(X_train):
    curr_clf = DecisionTreeClassifier(**best_params)
    curr_clf.fit(X_train[train_index], y_train[train_index])

    curr_pred = curr_clf.predict(X_test)
    predictions.append(curr_pred)

    curr_score = accuracy_score(y_test, curr_pred)
    scores.append(curr_score)



#%%   b. Train one Decision Tree on each subset, using the best hyperparameter values found above.
#     Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets,
#     these Decision Trees will likely perform worse than the first Decision Tree, achieving only
#     about 80% accuracy.

np.mean(scores) # 0.795

# tree_grid_clf.best_estimator_.get_params()

#%%   c. Now comes the magic. For each test set instance, generate the predictions of the 1,000
#     Decision Trees, and keep only the most frequent prediction (you can use SciPy’s mode()
#     function for this). This gives you majority-vote predictions over the test set.

#%%   d. Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than
#     your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random
#     Forest classifier!

