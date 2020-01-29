#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
import seaborn as sns

from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv('dane_dm/spam.dat')
print(df.info())
print(df['target'].value_counts())
#print(df.describe())
properties = list(df.columns.values)
properties.remove('target')
X = df[properties]
Y = df['target']
y = Y.replace(to_replace=['no', 'yes'], value=[0, 1])
#integer_mapping = {x: i for i,x in enumerate(Y)}
#y = np.array([integer_mapping[word] for word in Y])
#print(y)
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Default kernel accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Default linear kernel Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Default RBF kernel accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

svc=SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Default polynomial accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


