#!/usr/bin/env python
# coding: utf-8

# Let’s import the libraries that we are going to use for data manipulation, visualization, training the model, etc.
# import library to ploting
# import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import pandas as pd
# pd.set_option('display.float_format', lambda x: '%.4f' % x)
import numpy as np
# avoid alert
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
# Import library to preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
# performace metrics libray
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# select the GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU
# import library modules keras and tensorflow
from keras.layers import *
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Dense
from keras.layers import LSTM, GRU, Dropout, Bidirectional
import matplotlib.image  as mpimg
from keras import regularizers
from keras.callbacks import TensorBoard, ReduceLROnPlateau 
from keras import optimizers
import keras

def plot_prediction(y_train, y_train_inv, y_val, y_val_inv):
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_val)), y_val_inv.flatten(), marker='.', label="true")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_val)), y_pred_inv.flatten(), 'r', label="prediction")
    plt.ylabel('Throughput')
    plt.xlabel('Time Step')
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'prediction.eps'))
    plt.show();

def plot_prediction_detail(y_pred_inv, y_val_inv):
    plt.figure(figsize=(8,5))
    plt.plot(y_val_inv.flatten(), marker='.', label="true")
    plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
    plt.ylabel('Throughput')
    plt.xlabel('Time Step')
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'prediction_detail.eps'))
    plt.show();

# loss plot
def plot_history_metrics(history):
    plt.figure(figsize=(8,5))
    # plot metrics
    plt.plot(history.history['mean_absolute_error'], marker='.')
    plt.plot(history.history['mean_squared_error'], marker='.')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['mean_absolute_error', 'mean_squared_error'], loc='best')
    plt.savefig(os.path.join('./../plots', 'perf_metrics.eps'))
    plt.show()
    
def plot_history_loss(history):
    plt.figure(figsize=(8,5))
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    # loss=history.history['loss']
    # val_loss=history.history['val_loss']
    # epochs=range(len(loss)) # Get number of epochs
    
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='vaidation')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'loss.eps'))
    plt.show()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    # plt.plot(epochs, loss, val_loss)
    # plt.title('Training loss')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend(["Loss"])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# modeling my LSTM with API Keras:Bidirectional
def model_lstm(X_train):
    input_x = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Bidirectional(LSTM(units=128, activation='relu'))(input_x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1)(x)
    model = Model(inputs=input_x, outputs=x)
    return model

# fit model
def compile_fit(X_train, y_train, X_test, y_test):
    model = model_lstm(X_train)
    model.summary()
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=32, 
                        validation_data=(X_test, y_test), 
                        shuffle=False)
    return history, model
    
if __name__ == "__main__":

    # load the dataset
    dataset_throughput = pd.read_csv('../../file/dataset_throughput.csv', header=0)
    dataset_throughput.set_index('timestamp', inplace=True)
    dataset_throughput.sort_values('timestamp', inplace=True)

    # swap some columns positions
    columnsTitles = ['connect_time','request_ticks','year','month','hour','day','weekday','minute', 'iteration', 'second','delta_sys_time','delta_user_time','rate','received','delay','tcp_mean_wind','downthpt']
    dataset_throughput = dataset_throughput.reindex(columns=columnsTitles)

    # normalization
    dataset = dataset_throughput.iloc[:, 9:]
    dataset = dataset.astype('float')
    print(dataset.shape)
    TRAIN_SIZE = int(len(dataset) * 0.60) # 60% train set
    VALID_SIZE = int(len(dataset) * 0.80) # 20% valid and test set

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    second = dataset['second'].values
    second = np.reshape(second, (-1,1))
    second = onehot_encoder.fit_transform(second)

    # iteration = dataset['iteration'].values
    # iteration = np.reshape(iteration, (-1,1))
    # iteration = onehot_encoder.fit_transform(iteration)

    # dataset['iteration'] = iteration
    dataset['second'] = second

    dataset_train = dataset.iloc[:TRAIN_SIZE, :]
    print('\nTraining Set: ', dataset_train.shape)
    dataset_val = dataset.iloc[TRAIN_SIZE:VALID_SIZE, :]
    print('\nValidation Set: ', dataset_val.shape)
    dataset_test = dataset.iloc[VALID_SIZE:, :]
    print('\nTest Set: ', dataset_test.shape)

    # scaling the data --> normalization with downthpt!
    f_columns = ['second','delta_sys_time','delta_user_time','rate','received','delay','tcp_mean_wind','downthpt']

    f_transformer = RobustScaler()
    thrput_transformer = RobustScaler()

    f_transformer = f_transformer.fit(dataset_train[f_columns].to_numpy())
    thrput_transformer = thrput_transformer.fit(dataset_train[['downthpt']])

    dataset_train.loc[:, f_columns] = f_transformer.transform(dataset_train[f_columns].to_numpy())
    dataset_train['downthpt'] = thrput_transformer.transform(dataset_train[['downthpt']])

    dataset_val.loc[:, f_columns] = f_transformer.transform(dataset_val[f_columns].to_numpy())
    dataset_val['downthpt'] = thrput_transformer.transform(dataset_val[['downthpt']])

    dataset_test.loc[:, f_columns] = f_transformer.transform(dataset_test[f_columns].to_numpy())
    dataset_test['downthpt'] = thrput_transformer.transform(dataset_test[['downthpt']])
    
    # time steps --> look back
    time_steps = 11
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(dataset_train, dataset_train.downthpt, time_steps)
    X_test, y_test = create_dataset(dataset_val, dataset_val.downthpt, time_steps)
    X_val, y_val = create_dataset(dataset_test, dataset_test.downthpt, time_steps)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # train the model with train and validation set
    hist, model = compile_fit(X_train, y_train, X_test, y_test)

    # plot the model performence
    # plot_history_metrics(hist)
    plot_history_loss(hist)

    # make predictions whith test set with
    y_pred = model.predict(X_val)
    y_train_inv = thrput_transformer.inverse_transform(y_train.reshape(1, -1))
    y_val_inv = thrput_transformer.inverse_transform(y_val.reshape(1, -1))
    y_pred_inv = thrput_transformer.inverse_transform(y_pred)

    # plot the prediction performence
    plot_prediction(y_train, y_train_inv, y_val, y_val_inv)
    plot_prediction_detail(y_pred_inv, y_val_inv)
    print('\nFinished training!')
    
