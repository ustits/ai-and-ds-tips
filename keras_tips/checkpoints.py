# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

"""
Документация: https://keras.io/callbacks
"""

dataset = np.loadtxt('dataset/pima-indians-diabetes.data.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

X_train, X_test = np.split(X, 2)
Y_train, Y_test = np.split(Y, 2)

model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

"""
можно сохранять в несколько файлов
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
"""
filepath = "weights.best.hdf5"

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

"""
monitor - какое значение мониторить
mode='max' - означает какое значение мы перезаписываем, например:
  для val_acc должно быть быть max
  для val_loss должно быть min
"""
model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
"""
Заканчиваем тренировку модели, если перестает изменяться наблюдаемый параметр
patience - количество epoch без изменений, после которых надо остановиться
"""
early_stop_checkpoint = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                             verbose=1, mode='auto')

model.fit(X_train, Y_train, validation_split=0.3, epochs=50, batch_size=10,
          callbacks=[model_checkpoint, early_stop_checkpoint], verbose=1)

# model.load_weights(filepath)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


