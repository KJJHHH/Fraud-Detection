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
IMAGES_PATH = Path() / "images" / "Decision tree"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
raw_data = np.array(pd.read_csv(
    'C:/Users/USER/Desktop/Code/Project/分類/Credit_card_fraud_detection/creditcard.csv'))
X, y = raw_data[:, 1:30], raw_data[:, 30]
pca = PCA(n_components=10)
pca.fit(X)
X_pca = pca.transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=777, train_size=0.8)

import random
def find_range(X_train, y_train): # find x value not too impossible for y == 1
    for i in range(len(X_train[1])):
        X_k = []
        for k in range(len(X_train)):
            if X_train[:, i][k] >= min(X_train[:, i][y_train == 1]) and\
            X_train[:, i][k] <= max(X_train[:, i][y_train == 1]):
                X_k.append(X_train[:, i][k])
        if i == 0:
            X_maybe = np.array(X_k).reshape(len(X_k), 1)
        else:
            X_maybe = np.concatenate([X_maybe, np.array(X_k).reshape(len(X_k), 1)])
    
    return X_maybe

# X_maybe = find_range(X_train, y_train)
def create_batch(X_train, y_train): # for whole batch the number is too big
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

    return X_batch, y_batch

X_batch, y_batch = create_batch(X_train, y_train)
print(X_batch.shape, y_batch.shape)
'''print(f'rate:{len(X_1)/(len(X_0)+len(X_1))}')'''

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Voting
voting_clf = VotingClassifier( # Train
    estimators=[
        ('lr', LogisticRegression(max_iter = 1000, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(max_iter = 20, random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
for name, clf in voting_clf.named_estimators_.items():# Predict
    y_pred = clf.predict(X_test)
    print(name, "=", confusion_matrix(y_pred, y_test))

voting_clf.predict(X_test[:1])
print([clf.predict(X_test[:1]) for clf in voting_clf.estimators_])
y_pred = voting_clf.predict(X_test)
print(f'Hard Voting:\n{confusion_matrix(y_pred, y_test)}')


# Plot
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
save_fig("moons_polynomial_svc_plot")
plt.show()


# Soft voting
voting_clf.voting = "soft"
voting_clf.named_estimators["lr"].probability = True
voting_clf.named_estimators["rf"].probability = True
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_batch, y_batch)
voting_clf.predict(X_test)
print(f'Soft Voting:{confusion_matrix(y_pred, y_test)}')


# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

ranfor_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
ranfor_clf.fit(X_batch, y_batch)
y_pred = ranfor_clf.predict(X_test)
print(f'Bagging no sampling:\n{confusion_matrix(y_pred, y_test)}')

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, # out of bag sampling
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_batch, y_batch)
y_pred = bag_clf.predict(X_test)
print(f'Bagging out of bag (bootstrap sampling):{confusion_matrix(y_pred, y_test)}')

y_predall = [] # try several times out of bag
bag_num = 15
for i in range(bag_num): # odd number
    X_batch, y_batch = create_batch(X_train, y_train)
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
    bag_clf.fit(X_batch, y_batch)
    y_pred = bag_clf.predict(X_test)
    y_predall.append(y_pred)
    print(f'{i} round: \n {confusion_matrix(y_pred, y_test)}')
y_predall = np.array(y_predall).T

y_predlast = [] 
for i in range(len(y_predall[:, 1])): # 1 column
    for k in range(len(y_predall[1, :])): # 1 row

        z = 0
        o = 0
        if y_predall[i, k] == 0:
            z += 1
        if y_predall[i, k] == 1:
            o += 1 
    if z > o:
        y_predlast.append(0)
    if z < o:
        y_predlast.append(1)

print(f'all: \n {confusion_matrix(y_predlast, y_test)}')

from sklearn.ensemble import StackingClassifier


# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5  # number of cross-validation folds
)
stacking_clf.fit(X_batch, y_batch)
y_pred = stacking_clf.predict(X_test)

print(f'Stacking: \n {confusion_matrix(y_pred, y_test)}')