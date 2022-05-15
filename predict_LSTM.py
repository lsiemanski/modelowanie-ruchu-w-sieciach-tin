from cProfile import label
from turtle import width
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from matplotlib import pyplot as plt


look_back = 10
# X = np.load("X.npy")
Y = np.load("Y.npy")
Y = Y.reshape((-1))

X = [Y[i: look_back + i] for i in range(Y.shape[0] - look_back)]
X = np.array(X)
Y = Y[look_back:]

failure_time = 16000 - look_back

X_train = X[:failure_time, :]
X_test = X[failure_time:, :]
Y_train = Y[:failure_time]
Y_test = Y[failure_time:]

X_train = X_train.reshape(-1, look_back, 1)
X_test = X_test.reshape(-1, look_back, 1)

# chunk_size = 100
# chunks_number = int(X_test.shape[0] / chunk_size)

# X_train = np.array(np.array_split(X_train, chunks_number))
# Y_train = np.array(np.array_split(Y_train, chunks_number))
# X_test = np.array(np.array_split(X_test, chunks_number))
# Y_test = np.array(np.array_split(Y_test, chunks_number))

# Y_train = np.ravel(Y_train)
# Y_test = np.ravel(Y_test)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, Y_train, epochs=5)
prediction = model.predict(X_test)
prediction = prediction.reshape(-1)

print(prediction)
print(prediction.shape)
print(X_test.shape)

print(mean_absolute_percentage_error(Y_test, prediction))

x_axis = range(1000)
plt.plot(x_axis, Y_test[:1000], label='real', linewidth=0.5)
plt.plot(x_axis, prediction[:1000], label='predicted', linewidth=0.5)
plt.legend()
plt.show()
