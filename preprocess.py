import os
import pickle
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import h5py


train_dir = 'btc_data_extended/train_slices'
val_dir = 'btc_data_extended/val_slices'
scalers_dir = 'scalers_new'
weights_dir = 'pretrained_new'
slices = 36
x_len = 120
x_shape, y_shape = (x_len, 177), (3,)


def scale_data(col, scaler):
    if col.ndim == 1:
        col = col.reshape(-1, 1)
    res = scaler.transform(col)
    return res


def get_all_data(df, scalers, target='e_class', categorical=True):
    bin_indicators = ['volatility_bbhi', 'volatility_bbli', 
                  'volatility_kchi', 'volatility_kcli', 
                  'trend_psar_up_indicator', 'trend_psar_down_indicator']
    mon = to_categorical(df['mon'].values, 12)
    wd = to_categorical(df['wd'].values, 7)
    d = to_categorical(df['d'].values-1, 31)
    hr = to_categorical(df['hr'].values, 24)
    min_ = to_categorical(df['min'].values // 5, 12)
    X = np.concatenate([mon, wd, d, hr, min_], axis=1)
    for ind in bin_indicators:
        X = np.concatenate([X, df[ind].values.reshape(-1, 1)], axis=1)
    for key, sc in scalers.items():
        if key == 'ochl':
            res = scale_data(df[['o', 'c', 'h', 'l']].values, sc)
            X = np.concatenate([X, res], axis=1)
        else:
            res = scale_data(df[key].values, sc)
            X = np.concatenate([X, res], axis=1)
    if categorical:
        Y = to_categorical(df[target].values, df[target].nunique())
    else:
        Y = df[target].values
    return X, Y


def load_scalers(path=scalers_dir):
    scalers = {f[3:-4]: pickle.load(open(os.path.join(path, f),'rb')) for f in os.listdir(path)}
    return scalers


def get_class_weights(Y, x_len=x_len):
    class_weights = dict(zip([0,1,2],max(Y[x_len:].sum(axis=0)) / Y[x_len:].sum(axis=0)))
    print('class balance:', Y[x_len:].sum(axis=0))
    return class_weights


def get_datagen(X, Y, x_len=120, batch_size=64, shuffle=True):
    datagen = TimeseriesGenerator(X[1:], Y[:-1], x_len, batch_size=batch_size, shuffle=shuffle)
    x1, y1 = datagen.__getitem__(0)
    print('Data generating: ', x1.shape, y1.shape)

    return datagen