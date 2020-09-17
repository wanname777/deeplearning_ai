import matplotlib.pyplot as plt
import numpy as np


def plot_series(time, seties, format='-', is_show=True):
    plt.figure(figsize=(10, 6))
    plt.plot(time, seties, format)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.grid()
    if is_show:
        plt.show()


def trend(time, slope=0):
    # 类似斜率
    return slope * time


time = np.arange(4 * 365 + 1)
series = trend(time, 0.1)
plot_series(time, series)


def seasonal_pattren(season_time):
    # 等价三目运算符
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    # phase是p，amplitude是A，该函数周期T=period
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattren(season_time)


amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)

baseline = 10
slope = 0.05
series = baseline + trend(time, slope) + \
         seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


noise_level = 15
noisy_series = series + noise(time, noise_level, seed=42)
plot_series(time, noisy_series)


def autocorrelation1(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    fai1 = 0.5
    fai2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += fai1 * ar[step - 50]
        ar[step] += fai2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation2(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    fai = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += fai * ar[step - 1]
    return ar[1:] * amplitude


series = autocorrelation1(time, 10, seed=42)
plot_series(time[:200], series[:200])

series = autocorrelation2(time, 10, seed=42)
plot_series(time[:200], series[:200])

series = autocorrelation2(time, 10, seed=42) + \
         seasonality(time, period=50, amplitude=150) + \
         trend(time, 2)
series2 = autocorrelation2(time, 10, seed=42) + \
          seasonality(time, period=50, amplitude=2) + \
          trend(time, -1) + 550
series[200:] = series2[200:]
series += noise(time, 30)
plot_series(time[:300], series[:300])


def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=num_impulses)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


series = impulses(time, 10, seed=42)
plot_series(time, series)


def autocorrelation(source, fais):
    ar = source.copy()
    max_lag = len(fais)
    for step, value in enumerate(source):
        for lag, fai in fais.items():
            if step - lag > 0:
                ar[step] += fai * ar[step - lag]
    return ar


signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series, is_show=False)
plt.plot(time, signal, 'k-')
plt.show()

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series, is_show=False)
plt.plot(time, signal, 'k-')
plt.show()
