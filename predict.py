from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.load("X.npy")
Y = np.load("Y.npy")

failure_time = 16000

X_train = X[:failure_time, :]
X_test = X[failure_time:, :]
Y_train = Y[:failure_time, :]
Y_test = Y[failure_time:, :]

# Y_train = np.ravel(Y_train)
# Y_test = np.ravel(Y_test)

model = Sequential()
model.add(Dense(100))
model.add(Dense(Y_test.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train)
prediction = model.predict(X_test)

print(prediction)
print(prediction.shape)
print(Y_test.shape)

print(mean_absolute_percentage_error(Y_test, prediction))
