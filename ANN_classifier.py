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
df = pd.read_csv('dane_dm/spam.dat')
print(df.info())
print(df['target'].value_counts())
print(df.describe())
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

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(462,)),
    keras.layers.Dense(60, activation=tf.nn.relu),
	keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
