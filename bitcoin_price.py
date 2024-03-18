# Bit Predict - Regression
# https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import minmax_scale

df = pd.read_csv('Machine Learning 3/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv')

dates = df['Date']
prices = df['Closing Price (USD)']

# anytime plotter, start indx, end indx, skips for skipping inbetween values 
# def plotting(start:int, end:int, skips:int):
#     dates_ = []
#     prices_ = []
#     for i in range(start, end, skips):
#         dates_.append(dates[i])
#         prices_.append(prices[i])
#     plt.plot(dates_[:end], prices_[:end])
#     plt.xticks(rotation=90)
#     plt.show()

# plotting(start=0, end=500, skips=5)


# plot full based on years
# dates_ = []
# for date in dates:
#     dates_.append(datetime.strptime(date, '%Y-%m-%d'))
# plt.plot(dates_, prices)
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.show()


# Modelling : 

# 1. Naive Model

# split data
splitter = int(len(prices) * 0.8)
train_prices = prices[:splitter] # training prices
test_prices = prices[splitter:] # testing prices

X_train = train_prices[:-1] 
y_train = train_prices[1:]

X_test = test_prices[:-1]
y_test = test_prices[1:]

naive_model_score = mean_absolute_error(y_pred=y_test, y_true=X_test)


# 2. ANN Model Sliding Window (w=7, h=1)

WINDOW_SIZE = 7
HORIZON = 1

# create windows

df = pd.DataFrame()
df['prices'] = prices
df.index = dates
for i in range(WINDOW_SIZE):
    df[f'prices {i+1}'] = df['prices'].shift(i+1)
df.dropna(inplace=True)
df.rename(columns={'prices':'target'}, inplace=True)

# split data

X = df.drop(columns=['target'])
y = df['target']

splitter = int(0.8 * len(X))

X_train = X[:splitter]
y_train = y[:splitter]
X_test = X[splitter:]
y_test = y[splitter:]

ann_model_w7_h1 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='linear'),
    tf.keras.layers.Dense(units=32, activation='linear'),
    tf.keras.layers.Dense(units=32, activation='linear'),
    tf.keras.layers.Dense(units=1)
])
ann_model_w7_h1.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)
ann_model_w7_h1.fit(
    x=np.array(X_train),
    y=np.array(y_train),
    epochs=100,
    batch_size=64,
    shuffle=False,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1)
    ]
)
preds = ann_model_w7_h1.predict(X_test).flatten()
ann_model_w7_h1_score = mean_absolute_error(y_pred=preds, y_true=y_test)

# plot preds vs actuals
# plt.plot(y_test[:50], label='Actual')
# plt.plot(preds[:50], label='Predicted')
# plt.legend()
# plt.xticks(rotation=90)
# plt.show()

# 3. ANN Model Sliding Window (w=30, h=1)

df = pd.DataFrame()
df['prices'] = prices

WINDOW_SIZE = 30
HORIZON = 1

for i in range(WINDOW_SIZE):
    df[f'prices {i+1}'] = df['prices'].shift(i+1)
df.dropna(inplace=True)
df.rename(columns={'prices':'target'}, inplace=True)

X = df.drop(columns=['target'])
y = df['target']

splitter = int(0.8 * len(X))

X_train = X[:splitter]
y_train = y[:splitter]
X_test = X[splitter:]
y_test = y[splitter:]

ann_model_w30_h1 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='linear'),
    tf.keras.layers.Dense(units=32, activation='linear'),
    tf.keras.layers.Dense(units=1)
])
ann_model_w30_h1.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)
ann_model_w30_h1.fit(
    x=np.array(X_train),
    y=np.array(y_train),
    epochs=100,
    batch_size=64,
    shuffle=False,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2)
    ]
)
preds = ann_model_w30_h1.predict(X_test).flatten()
ann_model_w30_h1_score = mean_absolute_error(y_pred=preds, y_true=y_test)





# 4. CNN Model Sliding Window (w=7, h=1)

WINDOW_SIZE = 7
HORIZON = 1

df = pd.DataFrame()
df['prices'] = prices
df.index = dates
for i in range(WINDOW_SIZE):
    df[f'prices {i+1}'] = df['prices'].shift(i)
df.dropna(inplace=True)
df.rename(columns={'prices':'target'}, inplace=True)

X = df.drop(columns=['target'])
y = df['target']

splitter = int(0.8 * len(X))

X_train = X[:splitter]
y_train = y[:splitter]
X_test = X[splitter:]
y_test = y[splitter:]

cnn_model_w7_h1 = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1)),
    tf.keras.layers.Conv1D(filters=32, activation='relu', kernel_size=3, padding='causal'),  # same or causal
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=1)
])

cnn_model_w7_h1.compile(
    loss=tf.keras.losses.mae, 
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)

cnn_model_w7_h1.fit(
    x=np.array(X_train),
    y=np.array(y_train),
    verbose=2,
    batch_size=64, 
    epochs=100, 
    shuffle=False,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    ]
)

preds = cnn_model_w7_h1.predict(np.array(X_test)).flatten()
cnn_model_w7_h1_score = mean_absolute_error(y_pred=preds, y_true=np.array(y_test))





# 4. CNN Model Sliding Window (w=30, h=7)

WINDOW_SIZE = 30
HORIZON = 7

prices = list(prices)

i = 0
j = 30

X = []
y = []
list_ = []
while True:
    for a in range(i, j):
        list_.append(prices[a])
    X.append(list_)
    list_ = []
    for b in range(j, j+7):
        list_.append(prices[b])    
    y.append(list_)
    list_ = []
    i += 1
    j += 1
    if i == len(prices)-36:
        break

splitter = int(0.8 * len(prices))

X_train = X[:splitter]
X_test = X[splitter:]
y_train = y[:splitter]
y_test = y[splitter:]

cnn_model_w30_h7 = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1)),
    tf.keras.layers.Conv1D(filters=128, activation='linear', kernel_size=5, padding='causal'), # same or causal
    tf.keras.layers.Conv1D(filters=64, activation='linear', kernel_size=5, padding='same'), # same or causal
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=64, activation='linear'),
    tf.keras.layers.Dense(units=64, activation='linear'),
    tf.keras.layers.Dense(units=7)
])

cnn_model_w30_h7.compile(
    loss=tf.keras.losses.mae, 
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)

cnn_model_w30_h7.fit(
    x=np.array(X_train), 
    y=np.array(y_train),
    epochs=200, 
    verbose=2,
    shuffle=False,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    ]
)

preds = cnn_model_w30_h7.predict(np.array(X_test))
cnn_model_w30_h7_score = mean_absolute_error(y_pred=preds, y_true=np.array(y_test))



# 5. multivariate model CNN (w=7, h=1)

    # uni to multivariate, block size col

    # 28 November 2012 - 25mb
    # 9 July 2016 - 12.5mb
    # 18 May 2020 - 6.25mb

date_objs = []
for each in dates:
    date_objs.append(datetime.strptime(each, '%Y-%m-%d'))
block_sizes = []
for each in date_objs:
    if each < datetime(2012, 11, 28) and each < datetime(2016, 7, 9):
        block_sizes.append(25)
    elif each > datetime(2012, 11, 28) and each < datetime(2016, 7, 9):
        block_sizes.append(25)
    elif each >= datetime(2016, 7 , 9) and each < datetime(2020, 5, 18):
        block_sizes.append(12.5)
    elif each >= datetime(2020, 5, 18):
        block_sizes.append(6.25)

    # plot block size vs price vs dates
# plt.plot(dates, minmax_scale(block_sizes), label='block size')
# plt.plot(dates, minmax_scale(prices), label='prices')
# plt.xticks(rotation=90) 
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.legend()
# plt.show()

WINDOW_SIZE = 7
HORIZON = 1

df = pd.DataFrame()
df['prices'] = prices
df['block_size'] = block_sizes

for i in range(WINDOW_SIZE):
    df[f'prices {i+1}'] = df['prices'].shift(i)

df.dropna(inplace=True)
df.rename(columns={'prices':'target'}, inplace=True)

X = df.drop(columns=['target'])
y = df['target']

splitter = int(0.8 * len(X))

X_train = X[:splitter]
y_train = y[:splitter]
X_test = X[splitter:]
y_test = y[splitter:]

cnn_multi_model_w7_h1 = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=24, activation='linear'),
    tf.keras.layers.Dense(units=1)
])

cnn_multi_model_w7_h1.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)

cnn_multi_model_w7_h1.fit(
    x=np.array(X_train),
    y=np.array(y_train),
    epochs=100,
    verbose=2,
    batch_size=64,
    shuffle=False,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss'), tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='loss')]
)

preds = cnn_multi_model_w7_h1.predict(np.array(X_test)).flatten()

cnn_multi_model_w7_h1_score = mean_absolute_error(y_pred=preds, y_true=np.array(y_test))




# 6. NBeats (Neural Basis Expansion Analysis for Time Series Forecasting) too easy

WINDOW_SIZE = 7
HORIZON = 1

df = pd.DataFrame()
df['Target'] = prices
for i in range(WINDOW_SIZE):
    df[f'Price {i+1}'] = df['Target'].shift(i+1)

df.dropna(inplace=True)

X = df.drop(columns=['Target'])
y = df['Target']

splitter = int(0.8 * len(X))

X_train, X_test = X[:splitter], X[splitter:]
y_train, y_test = y[:splitter], y[splitter:]


# reasons to use Layer : handles shapes automatically, efficient-clean code, good integration of our architecture with the keras, use built-in functions.

class Nbeats(tf.keras.layers.Layer):
    def __init__(self, n_layers:int, n_neurons:int, theta_size:int, input_size:int):
            super().__init__() # needed to use features of Layer, to use its logic
            self.n_layers = n_layers,
            self.n_neurons = n_neurons,
            self.theta_size = theta_size,
            self.input_size = input_size
            self.dense_layers = [tf.keras.layers.Dense(units=n_neurons, activation='relu') for _ in range(n_layers)]
            self.theta_layer = tf.keras.layers.Dense(units=theta_size, activation='linear')
    def call(self, input_layer): # gets called whenever we pass an instance like layer1(input_layer)
        d_layer = input_layer
        for each in self.dense_layers:
           d_layer = each(d_layer) 
        output_layer = self.theta_layer(d_layer)
        backcast, forecast = output_layer[:, :7], output_layer[:, -1] # get first 7 vals and last elem from each row
        return backcast, forecast



layer1 = Nbeats(4, 32, 8, 7)
input_layer = tf.keras.layers.Input(shape=(7, ), dtype=tf.float32)

backcast, forecast = layer1(input_layer) # calls call()

# 30 stacks total
for _ in range(30):
    b, f = Nbeats(4, 32, 8, 7)(backcast)
    backcast = tf.subtract(backcast, b)
    forecast = tf.add(forecast, f)

nbeats_model = tf.keras.Model(inputs=input_layer, outputs=forecast)

nbeats_model.compile(
    loss=tf.keras.losses.mae, 
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mse']
)

nbeats_model.fit(
    x=X_train.to_numpy(), 
    y=y_train.to_numpy(),
    verbose=2,
    epochs=100,
    batch_size=128,
    shuffle=True,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3)]
)

preds = nbeats_model.predict(X_test.to_numpy())
nbeats_model_score = mean_absolute_error(y_pred=preds, y_true=y_test.to_numpy())

# 7. Ensemble Model (Homogenous)

WINDOW_SIZE = 7
HORIZON = 1

new_df = pd.DataFrame()
new_df['Target'] = prices
new_df.index = dates
for i in range(WINDOW_SIZE):
    new_df[f'Price {i+1}'] = new_df['Target'].shift(i+1)
new_df.dropna(inplace=True)

X = new_df.drop(columns=['Target'])
y = new_df['Target']

splitter = int(0.8 * len(X))

X_train = X[:splitter]
y_train = y[:splitter]
X_test = X[splitter:]
y_test = y[splitter:]


def create_models(n_models):
    ensemble_models = []
    for _ in range(n_models):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(7, ), dtype=tf.float32),
            tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=HORIZON, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.mae, 
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mse']
        )
        model.fit(
            x=X_train.to_numpy(),
            y=y_train.to_numpy(),
            epochs=1000,
            verbose=2,
            batch_size=128, 
            shuffle=True,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), tf.keras.callbacks.TensorBoard(log_dir='Machine Learning 3/logs/newone')] # for practise
        )
        ensemble_models.append(model)
    return ensemble_models

models = create_models(10)

preds = []
for model in models:
    preds.append(model.predict(X_test.to_numpy()).flatten())

i = 0

preds = np.array(preds)
ensemble_mean = np.mean(preds, axis=0)
ensemble_median = np.median(preds, axis=0)

ensemble_model_score = mean_absolute_error(y_pred=ensemble_mean, y_true=y_test.to_numpy())

# plot true and ensemble preds
# plt.plot(y_test)
# plt.plot(ensemble_mean)
# plt.xticks(rotation=90)
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20)) # intervals of 20 for x axis
# plt.show()

# plot intervals (ranges)
# std_d = np.std(preds, axis=0)
# intervals = std_d * 1.96
# upper = ensemble_mean + intervals
# lower = ensemble_mean - intervals

# plt.plot(y_test, label='Actual Values', color='lightblue')
# plt.plot(ensemble_mean, label='Predictions', color='yellow')
# plt.fill_between(
#     dates[-556:],
#     lower,
#     upper,
#     label='Intervals',
#     color='salmon'
# )
# plt.legend()
# plt.xticks(rotation=90)
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()






# PLOTTING ALL SCORES WHO WINS?

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(20, 10))
bars = ax.bar(range(0, 8), [naive_model_score, ann_model_w7_h1_score, ann_model_w30_h1_score, cnn_model_w7_h1_score, cnn_model_w30_h7_score, cnn_multi_model_w7_h1_score, nbeats_model_score, ensemble_model_score], color=['lightcoral', 'crimson', 'burlywood', 'violet', 'wheat', 'cornflowerblue', 'springgreen', 'lightskyblue'])
ax.set(
    xlabel='Models',
    ylabel='Scores (Lower is better)',
    title='Models vs Scores'
)
ax.set_xticks(range(0, 8), ['Naive Model', 'ANN_w7_h1', 'ANN_w30_h1', 'CNN_w7_h1', 'CNN_w_30_h7', 'CNN_w7_h1_mulvar', 'NBeats_w7_h1', 'Ensemble_w7_h1'])
ax.set_xticklabels(['Naive Model', 'ANN_w7_h1', 'ANN_w30_h1', 'CNN_w7_h1', 'CNN_w_30_h7', 'CNN_w7_h1_mulvar', 'NBeats_w7_h1', 'Ensemble_w7_h1'])
handles = []
for bar in bars:
    handles.append(Line2D([0], [0], color=bar.get_facecolor(), linewidth=5))
ax.legend(handles=handles, labels=['Naive Model', 'ANN_w7_h1', 'ANN_w30_h1', 'CNN_w7_h1', 'CNN_w_30_h7', 'CNN_w7_h1_mulvar', 'NBeats_w7_h1', 'Ensemble_w7_h1'])
plt.show()
