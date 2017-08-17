from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0],
              [0],
              [0],
              [1]], dtype=np.float32)

# wrapper for a neural network
model = Sequential()

# 1st Layer - Add an input layer of 32 nodes with the same input shape as
# the training samples in X
model.add(Dense(32, input_dim=X.shape[1]))

# 2rd Layer - Add a softmax activation layer
# convert linear results to probabilities
model.add(Activation('softmax'))

# 4th Layer - Add a fully connected output layer
model.add(Dense(1))

# 5th Layer - Add a sigmoid activation layer aka activation function
model.add(Activation('sigmoid'))
# equivalent to
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# "binary_crossentropy" here means that we are using only two classes

model.summary()

model.fit(X, y, epochs=1000, verbose=0)

model.evaluate()