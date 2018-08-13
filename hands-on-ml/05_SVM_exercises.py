#%% imports
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


# 8. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a SGDClassifier on
# the same dataset. See if you can get them to produce roughly the same model.

#%%
iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

#%% LinearSVC

lsvc_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])

lsvc_clf.fit(X, y)

lsvc_clf.named_steps['linear_svc'].coef_, lsvc_clf.named_steps['linear_svc'].intercept_

#%% SVC

svc_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1))
])

svc_clf.fit(X, y)

svc_clf.named_steps['svc'].coef_, svc_clf.named_steps['svc'].intercept_



#%% SGDClassifier

sgd_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(loss="hinge"))
])

sgd_clf.fit(X, y)

sgd_clf.named_steps['sgd'].coef_, sgd_clf.named_steps['sgd'].intercept_



#%% 9. Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary classifiers, you
# will need to use one-versus-all to classify all 10 digits. You may want to tune the
# hyperparameters using small validation sets to speed up the process. What accuracy can you
# reach?

mnist = datasets.fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])

ovr_clf = OneVsRestClassifier(svm_clf)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# use small subset for initial development
X_train, y_train = X_train[:500], y_train[:500]

#%%

ovr_clf.fit(X_train, y_train)


#%%
# try a prediction
predictions = ovr_clf.predict(X_test)
predictions


#%%

# get cross val score
# TODO see http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics

scores = cross_val_score(ovr_clf, X_train, y_train, cv=3)
'Accuracy: %{0:2f} (+/- %{1:2f})'.format(scores.mean(), scores.std() * 2)
# => 'Accuracy: %0.802210 (+/- %0.021665)'



