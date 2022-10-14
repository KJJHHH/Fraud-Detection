import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size = 14)
plt.rc('axes', labelsize = 14, titlesize = 14)
plt.rc('legend', fontsize = 14)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)

# Save fig
from pathlib import Path
IMAGES_PATH = Path() / "images" / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

raw_data = np.array(pd.read_csv(
    'C:/Users/USER/Desktop/Code/Project/分類/Credit_card_fraud_detection/creditcard.csv'))
X, y = raw_data[:, :30], raw_data[:, 30]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=777, train_size=0.8)


def iris_data():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris.target

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]
    return X, y

import random

X_0all = [] # majority
y_batch = []
for i in range(len(X_train)):
    if (y_train[i] == 1) == False:
        X_0all.append(X_train[i])
X_0 = np.array(random.sample(X_0all, k = 1500))
y_0 = np.zeros([len(X_0), 1])

X_1 = X_train[y_train == 1]
y_1 = np.ones([len(X_1), 1])

X_batch = np.concatenate([X_0, X_1])
y_batch = np.concatenate([y_0, y_1])
y_batch = y_batch.reshape(len(y_batch))
print(X_batch.shape, y_batch.shape)
print(f'rate:{len(X_1)/(len(X_0)+len(X_1))}')

# SVM Classifier model
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

# Models
svm_clf1 = make_pipeline(StandardScaler(), # Without kernel
                        LinearSVC(C=1, random_state=42)) 
svm_clf2 = SVC(kernel="linear", C=float("inf"), max_iter=10000) # Linear
svm_clf3 = SVC(kernel="linear", C=100, max_iter=10000) # Linear with C = 100
svm_clf4 = SVC(kernel="linear", C=100, max_iter=10000) # Linear with Standardized data

# NonLinear
polynomial_svm_clf = make_pipeline( # Polinomial without kernel
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10000, random_state=42))
poly_kernel_svm_clf = make_pipeline( # Polynomial with kernel
    StandardScaler(),
    SVC(kernel="poly", max_iter=10000, degree=3, coef0=1, C=5))
poly100_kernel_svm_clf = make_pipeline( # Polynomial with kernel
    StandardScaler(),
    SVC(kernel="poly", max_iter=10000, degree=10, coef0=100, C=5))
rbf_kernel_svm_clf = make_pipeline( # Rbf
    StandardScaler(),
    SVC(kernel="rbf", max_iter = 10000, gamma=5, C=0.001))

# rbfs
gamma1, gamma2 = 0.1, 5 # Rbf
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
rbf_kernel_svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", max_iter=10000, gamma=gamma, C=C)
    )
    rbf_kernel_svm_clfs.append(rbf_kernel_svm_clf)

all_svm_clfs = {'svm_clf1':svm_clf1, 'svm_clf2':svm_clf2, 'svm_clf3':svm_clf3, 'svm_clf4':svm_clf4, 
'poly100_kernel_svm_clf':poly100_kernel_svm_clf, 
'poly_kernel_svm_clf':poly_kernel_svm_clf, 'polynomial_svm_clf':polynomial_svm_clf}
s = 0
for i in rbf_kernel_svm_clfs:
    all_svm_clfs[f'rbf_kernel_svm_clf{s}'] = i
    s+=1

from sklearn.metrics import classification_report, confusion_matrix
for i in all_svm_clfs:
    all_svm_clfs[i].fit(X_batch, y_batch)
    y_pred = all_svm_clfs[i].predict(X_test)
    print(f'{i}:\n1. confusion_matrix:{confusion_matrix(y_test, y_pred)}')
    # print(f'2. classification_report:{classification_report(y_pred, y_train1)}')

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_

    plt.plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_up, "k--", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_down, "k--", linewidth=2, zorder=-2)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#AAA',
                zorder=-1)
