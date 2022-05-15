from cProfile import label
from turtle import width
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from matplotlib import pyplot as plt


def split_sequence(sequence, n_steps_in, n_steps_out):
    X = [sequence[i:i+n_steps_in]
         for i in range(len(sequence) - (n_steps_in + n_steps_out))]
    y = [sequence[i:i+n_steps_in]
         for i in range(n_steps_in, len(sequence) - n_steps_out)]

    return np.array(X), np.array(y)


look_back = 15
predict_forward = 10
take = 16000
Y = np.load("Y.npy")
Y = Y.reshape((-1))
Y = Y[:take]

X = [Y[i: look_back + i] for i in range(Y.shape[0] - look_back)]
X = np.array(X)


failure_time = 16000
train_test_split = 10000

Y = Y[look_back:]
Y = [Y[i: predict_forward + i] for i in range(Y.shape[0] - predict_forward)]
Y = np.array(Y)
X_train = X[:train_test_split, :]
X_test = X[train_test_split:, :]
Y_train = Y[:train_test_split, :]
Y_test = Y[train_test_split:, :]

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
model.add(Dense(predict_forward))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, Y_train, epochs=5)
prediction = model.predict(X_test)
# prediction = prediction.reshape(-1)

print(prediction)
print(prediction.shape)
print(X_test.shape)

Y_to_compare = [c[0] for c in Y_test]
Y_to_compare = np.concatenate((Y_to_compare, Y_test[-1, :]))
print(mean_absolute_percentage_error(Y_to_compare, prediction))


x_axis = range(50)
plt.plot(x_axis, Y_to_compare[:50], label='real', linewidth=0.5)
plt.plot(x_axis, prediction[0:50], label='predicted', linewidth=0.5)
plt.legend()
plt.show()
