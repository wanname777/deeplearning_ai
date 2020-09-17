import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def plot_series(time, series, format='-', start=0, end=None, is_show=True,
                is_fig=True):
    if is_fig:
        plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid()
    if is_show:
        plt.show()


file = pd.read_csv('Sunspots.csv', skiprows=1)
time = np.array(file.iloc[:, 0])
series = np.array(file.iloc[:, 2])
plot_series(time, series)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 60
batch_size = 100
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


dataset = windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60,
                           kernel_size=5,
                           strides=1,
                           padding='causal',
                           activation='relu',
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['mae'])
history = model.fit(dataset, epochs=500)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plot_series(time_valid, x_valid, is_show=False)
plot_series(time_valid, rnn_forecast, is_fig=False)

print(tf.keras.metrics.mse(x_valid, rnn_forecast).numpy())
print(tf.keras.metrics.mae(x_valid, rnn_forecast).numpy())

if __name__ == '__main__':
    print(1)
    print("nihao")
