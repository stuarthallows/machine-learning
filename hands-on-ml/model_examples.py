# linear SVM classification
# ============================

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

C = 5
alpha = 1 / (C * len(X))

# LinearSVC:
# regularises the bias term so center the training set first by subtracting it mean
# this is automatic if scaling with StandardScalar
# set the loss to 'hinge', it's not the default value
# for better performance set the 'dual' hyperparameter to False unless there are more features than training instances
# algorithm take longer if you require a very high precision, controller by the epsilon hyperparameter (tol in 
# SciKit-Learn), generally the default tolerance is fine.

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=C, loss="hinge", random_state=42))
])

svm_clf.fit(X, y)

# make predictions
svm_clf.predict([[5.5, 1.7]])


# alternatively use the following, however SVC is much slower, especially with large training sets
# SVC:
# Does support the kernel trick
# Gets very slow when the number of training instances gets large (e.g. hundreds of thousands of instances)
# Perfect for complex but small or medium sized training sets
# Scales well with the number of features, especially with sparse features
SVC(kernel="linear", C=C)

# another option uses regular Stochastic GD to train a linear SVM classifier, it does not converge
# as fast as LinearSVC but can handle large datasets that do not fit in memory (out-of-core
# training), or to handle online classification tasks.
SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha, n_iter=100000, random_state=42)



# Nonlinear SVM classification
# ============================

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# add polynomial to attempt to convert a non-linearly separable dataset into a linearly separable dataset
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(X, y)

# applying the kernel trick

# coef0 controls how much the model is influenced by high-degree polynomials versus low-degree polynomials
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X, y)


# Gaussian RBF kernel
# uses the kernel trick, making it possible to get the benefit of adding many similarity features, without actually 
# adding them

# Gamma acts like a regularisation hyperparameter
# 	If the model is overfitting reduce it
#	If the model is underfitting increase it

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)



# SVM linear regression
# =====================

from sklearn.svm import LinearSVR

# The width of the street is controlled by the ϵ hyperparameter
# training data should be scaled and centered first

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)



# SVM nonlinear regression
# ========================

from sklearn.svm import SVR

# use a small C value to apply regularisation
# SVR supports the kernel trick
# SVR gets much too slow when the training set grows large

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)



# Decision Trees
# ==============

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

# classification 
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# if less than a few thousand training instances set 'presort=True' to speed up training
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# generate a graph definition file
export_graphviz(
    tree_clf,
    out_file="images\\iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# use graphviz to visualise the definition file
# dot -Tpng iris_tree.dot -o iris_tree.png

# predictions 
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])


# regression
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)


# Hard voting classifier
# ============================

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# create and train a voting classifier composed of three diverse classifiers
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

# look at the accuracy of each - the voting classifier is best
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Soft voting classifier
# ============================

# As above but replace 'hard' with 'soft'
# If using SVC set 'probability=True' in order to add the predict_proba() method


# Bagging classifier
# ==================

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


# use BaggingRegressor for regressions


# Out-of-bag evaluation
# setting 'oob_score=True' requests an automatic oob evaluation after training

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

bag_clf.oob_decision_function_

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)


# Random Forests
# ==============

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)


# this bagging classifier is roughly ewuivalent, but prefer the random forest classifier
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
    )

# Extra-Trees
# ===========

# ExtraTreesClassifier - has same API as RandomForestClassifier
# ExtraTreesRegressor - has same API as RandomForestRegressor


# using a Randon Forest to detect feature importance

iris = load_iris()

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)


# AdaBoost classifier
# ===================

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5
    )

ada_clf.fit(X_train, y_train)

# also see AdaBoostRegressor

# Gradient Boosting regressor
# ===========================

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

gbrt.fit(X, y)



# Gradient Boosted Regression Trees ensembles
# ===========================================

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

gbrt.fit(X, y)


# use early stopping to find the optimal number of trees in a gradient boosting ensemble
# option 1 using staged_predict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

# staged_predict() returns an iterator over the predictions made by the ensemble at each stage of training 
# (with one tree, two trees, etc.)
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)


# use early stopping to find the optimal number of trees in a gradient boosting ensemble
# option 1 stop when the validation error does not improve for n iterations in a row

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break # early stopping



# Principal Component Analysis
# ============================

# calculate the first two PCs

X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]

# project a training set onto the plane defined by the first two PCs (using NumPy)

W2 = V.T[:, :2]
X2D = X_centered.dot(W2)

# project a training set onto the plane defined by the first two PCs (using Scikit-Learn)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

# having fit the PCA transformer access the PCs

pcs = pca.components_

# PCs stored as horizontal vectors, so the get the first PC

pc1 = pca.components_.T[:, 0]

# the proportion of dataset variance that lies along the axis of each PC

pca.explained_variance_ratio_


# choosing the right number of dimensions - option 1
# compute PCs without reducing dimensionality
# compute minimum number of dimensions required to preserve 95% variance
# run PCA again

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components = d)
X2D = pca.fit_transform(X)


# choosing the right number of dimensions - option 2
# indicate the ratio of variance to preserve

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

# choosing the right number of dimensions - option 3
# plot explained variance as a function of the number of dimensions (not shown)


# round tripping, compressing and decompressing data (a lossy operation)

pca = PCA(n_components = 154)
X_mnist_reduced = pca.fit_transform(X_mnist)
X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)


# applying incremental PCA - option 1
# good for large datasets or applying PCA online

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_mnist, n_batches):
    inc_pca.partial_fit(X_batch)

X_mnist_reduced = inc_pca.transform(X_mnist)


# applying incremental PCA - option 2

X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)


# applying randomised PCA

rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_mnist)


# performing kPCA with an rbf kernel

from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)


# selecting a kernel and tuning hyperparameters - option 1
# using kPCA to reduce dimensionality to 2 and then applt logistic regression
# find the best kernel and gamma value for kPCA to get the best accuracy
# best kernel and hyperparameters are then available through best_params_

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

grid_search.best_params_


# selecting a kernel and tuning hyperparameters - option 1
# select the kernel and hyperparameters that yield the lowest reconstruction error
# then use grid search with cross-validation to find the kernel and hyperparameters that minimize the pre-image
# reconstruction error

from sklearn.metrics import mean_squared_error

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)

X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# reconstruction pre-image error
mean_squared_error(X, X_preimage)


# Locally Linear Embedding
# ========================

from sklearn.manifold import LocallyLinearEmbedding

# a non-linear dimensionality reduction (NLDR) technique, particularly good at unrolling twisted manifolds, especially 
# when there is not too much noise

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)




