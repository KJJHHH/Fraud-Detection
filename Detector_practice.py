import numpy as np
import pandas as pd
from sklearn import tree
    # from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

Credit_card_data_pd = pd.read_csv("C:/Users/USER/Desktop/Kaggle/Credit_card_fraud_detection/creditcard.csv")
Credit_card_data_pd.shape
Credit_card_data = np.array(Credit_card_data_pd)
X = Credit_card_data[:, :30]
y = Credit_card_data[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

features = [i for i in Credit_card_data_pd][:30]
print(len(features))

# Decision Tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 5) # Build Classsifier
Detector_clf = clf.fit(X_train, y_train)

y_prediction = Detector_clf.predict(X_test) # Test
print(y_prediction)
print(y_test)

accuracy = metrics.accuracy_score(y_test, y_prediction) # Accuracy
print(accuracy)

export_graphviz(Detector_clf, out_file = 'tree.jpg', feature_names=features)# Visualise
forest = RandomForestClassifier(criterion = 'entropy', n_estimators=20, random_state=2, n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test) # Predict
accuracy = metrics.accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_pred, y_test))

# Random Forest
forest = RandomForestClassifier(criterion='entropy', n_estimators=20, random_state=2, n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test) # Predict
accuracy = metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_pred, y_test))

# SVM
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred))

# Logistic Regression
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_pred, y_test))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN:', confusion_matrix(y_pred, y_test))

# Two Layers Net
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import TensorDataset

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = self.softmax(self.h2o(x))

        return x

train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = Data.DataLoader(dataset=train_data, batch_size = 3000, shuffle=True, drop_last=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size = 3000, shuffle=True, drop_last=False)

def train(model, dataloader, epochs, criterion, optimizer, lr):
    for i in range(epochs):
        print(f'-------epochs:{i}---------')
        loss_history = []
        total_loss = 0
        acc_history = []
        acc_correct = 0
        train_len = 0
        for _, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model.forward(X)
            y_pred = y_pred.reshape(len(X), 2)
            loss = criterion(y_pred, y)
            total_loss += loss
            # print(y_pred)
            loss.backward()
            optimizer.step()
            
            for i in range(len(y_pred)):
                if y_pred[i, 0]>=y_pred[i, 1]:
                    y_pred_acc = 1
                else:
                    y_pred_acc = 0
                if y_pred_acc == y[i]:
                    acc_correct += 1
            train_len += len(y) # batch 

        acc = acc_correct/train_len
        print(f'total_loss:{total_loss}')
        print(f'acc:{acc}')
        loss_history.append(total_loss)
        acc_history.append(acc)

        total_loss_test, acc_test, y_pred_test = predict(model, test_loader)
        print(f'total_loss_test:{total_loss_test}')
        print(f'acc_test:{acc_test}')
    return total_loss, loss_history, y_pred_test

def predict(model, test_loader):
    total_loss = 0
    test_len = 0
    acc_correct = 0
    y_pred_ = []
    for _, (X, y) in enumerate(test_loader):

        y_pred = model.forward(X)
        loss = criterion(y_pred, y)
        total_loss += loss

        for i in range(len(y_pred)):
            if y_pred[i, 0]>=y_pred[i, 1]:
                y_pred_acc = 1
                y_pred_.append(y_pred_acc)
            else:
                y_pred_acc = 0
                y_pred_.append(y_pred_acc)

            if y_pred_acc == y[i]:
                acc_correct += 1
        test_len += len(y) # batch = 3000

    
    acc = acc_correct/test_len


    return total_loss, acc, y_pred_

model = Net(30, 20, 2)
epochs = 5
lr = .005
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = lr)
total_loss, loss_history, y_pred_test = train(model, train_loader, epochs, criterion, optimizer, lr)
print('-------Two Layer Net-------')
print(f'Classification report:{classification_report(y_test, y_pred)}')
print(f'confusion matrix:{confusion_matrix(y_pred, y_test)}')

# Two Layers with sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

y_train_dataframe = pd.DataFrame(y_train, columns=['expect']) # y_train encoding
y_train_encoding = pd.get_dummies(y_train_dataframe['expect'])

def TwoLayerForward():
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=30))
    clf.add(Dense(2, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf


clf = KerasClassifier(TwoLayerForward, 
    epochs = 50, batch_size = 500, verbose = 0) # make keras model compatible with sklearn

clf.fit(X_train, y_train_encoding)

# Ensemble
from sklearn.model_selection import cross_validate
from sklearn import ensemble, preprocessing, metrics
from sklearn.naive_bayes import GaussianNB
import mlxtend
import xgboost as xgb

# bagging
bag = ensemble.BaggingClassifier(n_estimators=10)
fit = bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
print('-------bagging-------')
print(f'Classification report:{classification_report(y_test, y_pred)}')
print(f'confusion matrix:{confusion_matrix(y_pred, y_test)}')

# Adaboost
boost = ensemble.AdaBoostClassifier(n_estimators=10)
boost.fit(X_train, y_train)
y_pred = boost.predict(X_test)
print('-------Adaboost-------')
print(f'Classification report:{classification_report(y_test, y_pred)}')
print(f'confusion matrix:{confusion_matrix(y_pred, y_test)}')

# XGBoost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print('-------XGBoost-------')
print(f'Classification report:{classification_report(y_test, y_pred)}')
print(f'confusion matrix:{confusion_matrix(y_pred, y_test)}')

# Stacking
import mlxtend.classifier as mlclass
from sklearn.linear_model import LogisticRegression

clf1 = KNeighborsClassifier(n_neighbors=9)
clf2 = RandomForestClassifier(criterion='entropy', n_estimators=20, random_state=2, n_jobs=2)
clf3 = GaussianNB()
clf4 = linear_model.LogisticRegression()
clf5 = KerasClassifier(TwoLayerForward, epochs = 50, batch_size = 500, verbose = 0)
clf6 = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
estimators = (('KNN', clf1), ('Random Forest', clf2), ('GaussianNB', clf3))#, clf4, clf5, clf6'
lr = LogisticRegression()
sclf = mlclass.StackingClassifier(classifiers=estimators, meta_classifier=lr)



