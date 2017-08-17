import numpy as np
from keras.utils import np_utils
import tensorflow as tf

# tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype('float32')
y = np.array([[0], [1], [1], [0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Building the model
xor = Sequential()

# split from [0] to [1, 0]
# e.g. converts vector to binary class matrix
y = np_utils.to_categorical(y)

print(y)

# Add required layers
# output nodes = 32, inputDim = 2
xor.add(Dense(32, input_dim=X.shape[1]))
xor.add(Activation('tanh'))
xor.add(Dense(2))
xor.add(Activation('sigmoid'))
# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, epochs=500, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
