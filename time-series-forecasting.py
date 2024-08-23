# time series are ordered sequence of values, usually equally spaced
# can be univariate or multivariate
# trend: a continuous up/down trend
# seasonality: patterns repeat at given interval- peaks and troughs
# some TS have both trend & seasonality while others are just white noise

# fixed partitioning: split the TS into training, validation and test periods
# TS train on test set as well
# Roll-forward partitioning: start with a short rolling period & gradually increase
# it by 1 day, week so on at a time
# we train the model on the training period and use it to forecast the following day/week
# in the validation period
# roll-forward mimics production conditions but require more production time
# metric: (root) mean square error / mean absolute error, mean absolute percentage error (mape)
# moving average: mean of n last values - eliminates a lot of the noise but
# does not anticipate seasonality/trend requiring differencing to remove them
# differencing: remove trend & seasonality; e.g, study values at time t and t-365 earlier:
# forecast = moving average of differenced series + series(t-365)
# to improve the forecast, smooth both past and present values:
# forecast = trailing moving average of differenced series + centered moving average of past series(t-365)
# moving average of centered windows more accurate than trailing windows
# we can't use centered windows to smooth present values as we don't know future values.


# 1. Naive forcasting
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

plt.show()
def plot_series(time, series, format = '_', start = 0, end = None, label = None):
    plt.plot(time[start:end], series[start:end], format, label = label, linestyle = '-')
    plt.xlabel('Time')
    plt.ylabel('Value')
    if label:
        plt.legend(fontsize = 14)
    plt.grid(True)

def trend(time, slope = 0):
    return slope * time

def seasonal_pattern(season_time):
    """just an arbitrary pattern"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2), 1 / np.exp(season_time * 3))


def seasonality(time, period, amplitude = 1, phase=0):
    """repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

# Trend and seasonality
time = np.arange(4 * 365 + 1)
slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
noise_level = 5
noise = white_noise(time, noise_level=noise_level, seed=42)
series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
# try naive forecast
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
naive_forecast = series[:split_time - 1: -1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label = 'series')
plot_series(time_valid, naive_forecast, label = 'forecast')
plt.show()
# zoom in the start of the validation period
# you see the naive forecast lagging 1 step behind the time series
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start = 0, end = 150, label = 'series')
plot_series(time_valid, naive_forecast, start = 1, end = 151, label = 'forecast')
plt.show()
# mean absolute error
errors = naive_forecast - x_valid
abs_errors = np.abs(errors)
mae = np.mean(abs_errors) # or abs_errors.mean()
mae

# Moving averages
def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)

def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast
     This implementation is *much* faster than the previous one"""
  mov = np.cumsum(series)
  mov[window_size:] = mov[window_size:] - mov[:-window_size]
  return mov[window_size - 1:-1] / window_size

moving_avg = moving_average_forecast(series, window_size = 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, moving_avg, label="Moving average (30 days)")
plt.show()

errors = moving_avg - x_valid
mae = np.mean(abs_errors)
# remove trend and seasonality to improve moving average
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series, label="Series(t) – Series(t–365)")
plt.show()
# focus on validation period
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) – Series(t–365)")
plt.show()

# moving avg on differenced series
diff_m_avg = moving_average_forecast(diff_series, window_size = 50)[split_time - 365 - 50:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label = "series(t) - series(t-365)")
plot_series(time_valid, diff_m_avg, label = "Moving average of diff")
plt.show()

# bring back trend & seasonality
diff_m_avg_plus_past = series[split_time - 365:-365] + diff_m_avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="series")
plot_series(time_valid, diff_m_avg_plus_past, label="forecast")
plt.show()

errors = diff_m_avg - x_valid
np.mean(np.abs(errors))

# use moving average on past values to remove some of the noise
diff_m_avg_plus_smooth_past = (moving_average_forecast(series[split_time - 370:-359], 11) +
                               diff_m_avg)
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_m_avg_plus_smooth_past, label="Forecasts")
plt.show()

# forecasting with machine learning
# train a model to forecast the next step given the last 30 steps
# first create a dataset of 30-step windows for training

def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1],window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# linear model
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[window_size])
])
model.compile(loss=keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])

model.fit(train_set, epochs=100, validation_data=valid_set)