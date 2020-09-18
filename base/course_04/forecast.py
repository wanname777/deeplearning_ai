import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


# from tensorflow import keras


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


def trend(time, slope=0):
    # 类似斜率
    return slope * time


def seasonal_pattren(season_time):
    # 等价三目运算符
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    # phase是p，amplitude是A，该函数周期T=period
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattren(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype='float32')
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
series = baseline + trend(time, slope) + \
         seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)
plot_series(time, series)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plot_series(time_train, x_train)
plot_series(time_valid, x_valid)

naive_forecast = series[split_time - 1:-1]
plot_series(time_valid, x_valid, is_show=False)
plot_series(time_valid, naive_forecast, is_fig=False)

plot_series(time_valid, x_valid, start=0, end=150, is_fig=True, is_show=False)
plot_series(time_valid, naive_forecast, start=1, end=151, is_fig=False)

print(tf.keras.metrics.mse(x_valid, naive_forecast).numpy())
# print(tf.keras.metrics.mean_squared_error(x_valid,naive_forecast).numpy())
print(tf.keras.metrics.mae(x_valid, naive_forecast).numpy())


# print(tf.keras.metrics.mean_absolute_error(x_valid,naive_forecast).numpy())


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
diff_moving_avg = moving_average_forecast(
    diff_series, 50)[split_time - 365 - 50:]
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
diff_moving_avg_plus_smooth_past = moving_average_forecast(
    series[split_time - 370:-360], 10) + diff_moving_avg
plot_series(time_valid, x_valid, is_show=False)
plot_series(time_valid, diff_moving_avg_plus_smooth_past, is_fig=False)

print(tf.keras.metrics.mse(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(tf.keras.metrics.mae(x_valid, diff_moving_avg_plus_smooth_past).numpy())

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


dataset = windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size)
# l0=tf.keras.layers.Dense(1,input_shape=[window_size])
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(
    lr=7e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)
# print(l0.get_weights())

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

plot_series(time_valid, x_valid, is_show=False)
plot_series(time_valid, results, is_fig=False)

print(tf.keras.metrics.mse(x_valid, results).numpy())
print(tf.keras.metrics.mae(x_valid, results).numpy())

