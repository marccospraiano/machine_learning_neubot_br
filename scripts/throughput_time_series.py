#!/usr/bin/env python
# coding: utf-8

# Let’s import the libraries that we are going to use for data manipulation, visualization, training the model, etc.
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from matplotlib import rc
import seaborn as sns

import pandas as pd
from math import sqrt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy import stats

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU

import keras
from tensorflow.keras import backend as K
from keras.layers import *
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import MaxPool1D, Activation, Conv1D, Conv2D, MaxPool2D, Flatten, TimeDistributed
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, BatchNormalization
from keras.models import Model, Input, Sequential
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras import optimizers

# from tensorflow.keras.models import Model, model_from_json
# from tensorflow.keras.layers import Input, GRU, Conv2D, Dropout, Flatten, Dense, Reshape, Concatenate, Add
# from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import TensorBoard


def plot_prediction_window(train, train_inv, test, test_inv, pred_inv):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(train)), train_inv.flatten(), 'g', label="History")
    plt.plot(np.arange(len(train), len(train) + len(test)), test_inv.flatten(), marker='.', label="True")
    plt.plot(np.arange(len(train), len(train) + len(test)), pred_inv.flatten(), 'r', label="Prediction")
    plt.ylabel('Throughput')
    plt.xlabel('Time Step')
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'prediction.jpg'))
    plt.grid(True)
    plt.show();

    
def plot_prediction_downthpt(pred_inv, test_inv):
    plt.figure(figsize=(10, 6))
    plt.plot(test_inv.flatten(), marker='.', label="True")
    plt.plot(pred_inv.flatten(), 'r', label="Prediction")
    plt.ylabel('Throughput')
    plt.xlabel('Time Step')
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'prediction_detail.jpg'))
    plt.grid(True)
    plt.show();


def plot_mse(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mean_squared_error'], marker='.')
    plt.plot(history.history['val_mean_squared_error'], marker='.')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join('./../plots', 'mse.eps'))
    plt.grid(True)
    plt.show()

    
def plot_mae(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mean_absolute_error'], marker='.')
    plt.plot(history.history['val_mean_absolute_error'], marker='.')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join('./../plots', 'mae.eps'))
    plt.grid(True)
    plt.show()
    
    
def plot_loss_zoom(history):
    plt.figure(figsize=(10, 6))
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    # epochs=range(len(loss)) # Get number of epochs
    
    zoomed_train_loss = loss[40:88]
    zoomed_val_loss = val_loss[40:88]
    zoomed_epochs = range(40,88)
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(zoomed_epochs, zoomed_train_loss, label='Training loss')
    plt.plot(zoomed_epochs, zoomed_val_loss, label='Validation loss')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")    
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'loss_zoom.eps'))
    plt.grid(True)
    plt.show()
    
    
def plot_history(history):
    plt.figure(figsize=(10, 6))
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    # loss=history.history['loss']
    # val_loss=history.history['val_loss']
    # epochs=range(len(loss)) # Get number of epochs
    
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join('./../plots', 'loss.eps'))
    plt.grid(True)
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
        v = X[i:(i + time_steps)]
        Xs.append(v)        
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def normalization(rawdata, normalise=2):
    rows, cols = rawdata.shape
    data = np.zeros(rawdata.shape)
    scale = np.ones(cols, dtype='float32')
    if normalise == 0: # do not normalise
        data = rawdata
        
    if normalise == 1: # same normalisation for all timeseries
        data = rawdata / np.max(rawdata)
        
    if normalise == 2: # normalise each timeseries alone. This is the default mode
        for i in range(cols):
            scale[i] = np.max(np.abs(rawdata[:, i]))
            data[:, i] = rawdata[:, i] / scale[i]
    return data

"""
def create_model(train):
    input_x = Input(shape=(train.shape[1], train.shape[2]))
    x = Conv1D(filters=100, kernel_size=2, strides=1)(input_x)
    x = Conv1D(filters=100, kernel_size=2, strides=1)(input_x)
    x = Dropout(rate=0.2)(x)
    x = Activation('relu')(x)
    x = Bidirectional(GRU(100, kernel_initializer='glorot_uniform'))(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=1)(x)
    model = Model(inputs=input_x, outputs=x)
    return model
    """


def create_model(train):
    input_x = Input(shape=(train.shape[1], train.shape[2]))
    x = Conv1D(filters=64, kernel_size=2, strides=1)(input_x)
    x = Activation('relu')(x)
    x = GRU(units=64, kernel_initializer='RandomNormal', return_sequences=True)(input_x)
    x = Activation('relu')(x)
    x = GRU(units=64, kernel_initializer='RandomNormal', return_sequences=False)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=1)(x)
    model = Model(inputs=input_x, outputs=x)
    return model


def compile_fit(train, target_train, test, target_test):
    EPOCHS = 30 # previous = 40
    BATCH_SIZE = 64
    model = create_model(train)
    model.summary()
    sgd_0 = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9) # previous lr=0.05, decay=1e-5, momentum=0.9)
    sgd_1 = optimizers.SGD(lr=0.5, decay=0, nesterov=True)
    sgd_2 = optimizers.Adam(learning_rate=0.001)
    # model.compile(loss=tf.keras.losses.Huber(), optimizer=sgd_1, metrics=['mean_absolute_error', 'mean_squared_error'])
    model.compile(loss='mean_absolute_error', optimizer=sgd_2, metrics=['mean_absolute_error', 'mean_squared_error'])
    # reduce_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=80, verbose=1, mode='auto', min_lr=0.001)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0, 
                               patience=25, # previous = 30
                               verbose=1, 
                               mode='max',
                               baseline=None,
                               restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('./../output_files/model_{epoch:02d}-{val_loss:.2f}.h5', 
                                       monitor='val_loss',verbose=1, save_best_only=True,
                                       save_weights_only=False, mode='auto')
    history = model.fit(train, target_train,
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(test, target_test), 
                        shuffle=False,
                        callbacks=[early_stop, model_checkpoint],
                        verbose=1)
    return history, model


def one_hot_encoder(df, col):
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    for c in col:        
        v = df[c].values
        v = np.reshape(v, (-1,1))
        v = onehot_encoder.fit_transform(v)
        df[c] = v
    return df
    
    
if __name__ == "__main__":

    # load the dataset
    dataset_throughput = pd.read_csv('../../file/dataset_throughput.csv', header=0)
    dataset_throughput.set_index('timestamp', inplace=True)
    # dataset_throughput.sort_values('timestamp', inplace=True)
    
    ## Mutual Information-Regression ##
    columnsTitles = ['year',
                     'connect_time',
                     'delay',
                     'hour',
                     'minute',
                     'month',
                     'request_ticks',
                     'day',
                     'second',
                     'weekday',
                     'iteration',
                     'delta_sys_time',
                     'tcp_mean_wind',
                     'rate',
                     'received',
                     'delta_user_time',
                     'downthpt']
    dataset_throughput = dataset_throughput.reindex(columns=columnsTitles)
    dataset = dataset_throughput.iloc[:, 8:].values
    # print(dataset.columns)
    dataset = dataset.astype('float32')
    print(dataset.shape)
    TRAIN_SIZE = int(len(dataset) * 0.60) # 60% train set
    VALID_SIZE = int(len(dataset) * 0.80) # 20% valid and test set

    # here we going to normalize the categorical features with One Hot Encoder
    # col = ['second','weekday','iteration']
    # dataset = one_hot_encoder(dataset, col)
    data = normalization(dataset, 2)
    dataset_train = data[:TRAIN_SIZE, :]
    print('\nTraining Set: ', dataset_train.shape)

    dataset_val = data[TRAIN_SIZE:VALID_SIZE, :]
    print('\nValidation Set: ', dataset_val.shape)

    dataset_test = data[VALID_SIZE:, :]
    print('\nTest Set: ', dataset_test.shape)

    # We are going to use StandardScaler from sklearn library to scale the data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    # train_arr = scaler.fit_transform(dataset_train)
    # val_arr = scaler.transform(dataset_val)
    # test_arr = scaler.transform(dataset_test)
    
    """
    # We are going to use StandardScaler from sklearn library to scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    train_arr = scaler.fit_transform(dataset_train)
    val_arr = scaler.transform(dataset_val)
    test_arr = scaler.transform(dataset_test)
    # split into input and outputs
    train_X, train_y = train_arr[:, :-1], train_arr[:, -1]
    val_X, val_y = val_arr[:, :-1], val_arr[:, -1]
    test_X, test_y = test_arr[:, :-1], test_arr[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)
    """
    
    """
    # normalization without downthpt!
    f_columns = ['second',
                 'weekday',
                 'iteration',
                 'delta_sys_time',
                 'tcp_mean_wind',
                 'rate',
                 'received',
                 'delta_user_time']
    scaler_transf = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    # f_transformer = StandardScaler()
    # thrput_transformer = StandardScaler()

    scaler_transf = scaler_transf.fit(dataset_train[f_columns].to_numpy())
    scaler_target = scaler_target.fit(dataset_train[['downthpt']])

    dataset_train.loc[:, f_columns] = scaler_transf.transform(dataset_train[f_columns].to_numpy())
    dataset_train['downthpt'] = scaler_target.transform(dataset_train[['downthpt']])

    dataset_val.loc[:, f_columns] = scaler_transf.transform(dataset_val[f_columns].to_numpy())
    dataset_val['downthpt'] = scaler_target.transform(dataset_val[['downthpt']])

    dataset_test.loc[:, f_columns] = scaler_transf.transform(dataset_test[f_columns].to_numpy())
    dataset_test['downthpt'] = scaler_target.transform(dataset_test[['downthpt']])
    print('\nFinished normalization...')
    
    train_arr = dataset_train.values
    val_arr = dataset_val.values
    test_arr = dataset_test.values
    """
    
    print('\nSaving test dataset...')
    # dataset_test.to_csv('../../file/dataset_test.csv', encoding='utf-8', index=False)
    np.savetxt('../../file/dataset_test.csv', dataset_test, delimiter=",")

    time_steps = 5
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(dataset_train, dataset_train[:, -1], time_steps)
    X_val, y_val = create_dataset(dataset_val, dataset_val[:, -1], time_steps)
    X_test, y_test = create_dataset(dataset_test, dataset_test[:, -1], time_steps)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    # build the model: 
    print('\nBuild GRU model...')
    hist, model = compile_fit(X_train, y_train, X_val, y_val)

    # plot the model performence
    print('\nPlots...')
    plot_history(hist)
    plot_loss_zoom(hist)
    plot_mae(hist)
    plot_mse(hist)
    
    # make predictions whith test set with
    print('\nMaking Predictions...')
    y_pred = model.predict(X_test)
    y_train_inv = scaler_target.inverse_transform(y_train.reshape(1,-1))
    y_test_inv = scaler_target.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    
    # calculate MAE
    mae = sqrt(mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten()))
    print('\nTest MAE: %.3f' % mae)
        
    # Root Mean Squared Error
    rmse = sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
    print('\nTest RMSE: %.3f' % rmse)

    # plot the prediction performance
    plot_prediction_window(y_train, y_train_inv, y_test, y_test_inv, y_pred_inv);
    plot_prediction_downthpt(y_pred_inv, y_test_inv)
    
    model.save('./../output_file/train_model.h5')  # creates a HDF5 file 'model_train.h5'
    del model  # deletes the existing model
    print("\nSaved model...")
    # model.save_weights('./../output_files/model_lstm_train.h5')
    print('\nFinished training...')