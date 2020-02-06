import pandas as pd                                     # Used to treat dataframes
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense

import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

dfPoints = pd.read_csv("df_points.txt", delimiter="\t") # Import dataset

dfPoints = dfPoints.drop(['Unnamed: 0'], axis=1)        # Remove the ID column

# Divide columns from dataset in dependent and independent variables
independent_variables = dfPoints[['x', 'y', 'z']]
dependent_variable = dfPoints['label']

# Divide dataset in training and test samples
X_train,X_test,y_train,y_test = train_test_split(independent_variables,dependent_variable,test_size=0.10,random_state=0)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
y_train = y_train.reshape(9000, 1)

BATCH_SIZE = 32
model.fit(x=X_train, y=y_train, epochs=5, steps_per_epoch=9000) 