from array import array
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

# Decision tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
# Regularization hyperparameters
tree_clf2 = DecisionTreeClassifier(min_samples_leaf = 10, max_depth=10, random_state=42)
tree_clf.fit(X_train, y_train)
tree_clf2.fit(X_train, y_train)

# Draw tree
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf2,
        out_file=str(IMAGES_PATH / "fraud_tree.dot"),  # path differs in the book
        feature_names=[f"{i}" for i in range(30)],
        class_names=['a', 'b'],
        rounded=True,
        filled=True
    )

from graphviz import Source

Source.from_file(IMAGES_PATH / "fraud_tree.dot")  # path differs in the book

# Estimate prob
tree_clf.predict_proba(y_test).round(3)
tree_clf.predict(y_test)