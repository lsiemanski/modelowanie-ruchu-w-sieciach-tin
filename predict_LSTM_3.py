from cProfile import label
from turtle import width
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    i_range = len(data) - look_back - 1
    print(i_range)
    for i in range(0, i_range):
        # index can move down to len(dataset)-1
        dataX.append(data[i:(i+look_back)])
        # Y is the item that skips look_back number of items
        dataY.append(data[i + look_back])

    return np.array(dataX), np.array(dataY)


look_back = 5
take = 16000
Y = np.load("Y.npy")
# Y = Y[:take]
data_range = (-1, 1)
scaler = MinMaxScaler(feature_range=data_range)

Y = scaler.fit_transform(Y)

failure_time = 16000
train_test_split = 10000

train_size = int(len(Y) * 0.67)
test_size = len(Y) - train_size
train, test = Y[0:train_size, :], Y[train_size:len(Y), :]
print(len(train), len(test))


# X_train is input, Y_train is expected output
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)


print("Original X_train shape:", X_train.shape)
# timestep = 1, input_dim = X_train.shape[1]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))
print("New X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Y_train example:", Y_train[0])


model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, Y_train, epochs=5)
prediction = model.predict(X_test)
prediction = prediction.reshape(-1)

print(prediction)
print(prediction.shape)
print(X_test.shape)

print(mean_absolute_percentage_error(Y_test, prediction))

x_axis = range(50)
plt.plot(x_axis, Y_test[:50], label='real', linewidth=0.5)
plt.plot(x_axis, prediction[:50], label='predicted', linewidth=0.5)
plt.legend()
plt.show()
