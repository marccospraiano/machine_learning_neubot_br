# import library to ploting
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
# %matplotlib inline
import urllib.request
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import numpy as np
# avoid alert
import warnings
warnings.filterwarnings('ignore')
# import library to preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pandas.io.json import json_normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
import requests
import locale
import json, tempfile
import math
# performace metrics libray
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# select the GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU
from util import *
# import library modules keras and tensorflow
from keras.layers import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Dropout
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.image  as mpimg
from keras.models import Model, Input
from keras import regularizers
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

"""
def load_dataset(file2015, file2016, file2017, file2018):
    base_2015 = pd.read_json(file2015)
    print('Base 2015 => ', base_2015.shape)
    base_2016 = pd.read_json(file2016)
    print('Base 2016 => ', base_2016.shape)
    base_2017 = pd.read_json(file2017)
    print('Base 2017 => ', base_2017.shape)
    base_2018 = pd.read_json(file2018)
    print('Base 2018 => ', base_2018.shape)
    return base_2015, base_2016, base_2017, base_2018
    """

def load_dataset(file2018):
    base_2018 = pd.read_csv(file2018)
    print('Base 2018 => ', base_2018.shape)
    return base_2018

def drop_col_values_missing(dataset):
    # dataset = dataset.drop(['AS_name','_id','city','elapsed_target','internal_address','platform','real_address','remote_address','version',],axis=1)
    dataset.dropna(inplace=True)
    print(dataset.isna().sum())
    print(dataset.shape)
    return dataset

def drop_not_year(base, year): 
    # base.sort_values(['timestamp', 'iteration', 'uuid'], inplace=True, ascending=True)
    indice = base[base['timestamp'].dt.year != year].index
    base.drop(indice, inplace=True)
    base = base.reset_index(drop=True)
    print('Last shape:\n', base.shape)
    return base

def cria_sessao(df):    
    # cria Session
    print("creating session...")
    df.sort_values(['uuid','timestamp','iteration'], inplace=True)
    df['session'] = df['iteration']

    val = df.uuid.value_counts()
    val = val.index

    cont = -1
    aux = df[df['uuid'] == val[0]]
    x2 = aux.index
    x2 = x2[0]    
    for i in val:
        aux = df[df['uuid'] == i]
        for x in aux.index:
            if df.iteration.loc[x] <= df.iteration.loc[x2]:
                cont += 1
            df.session.loc[x] = cont
            x2 = x
    return df

def drop_col_values_missing_another(dataset):
    dataset = dataset.drop(['AS_name','_id','city',
                                  'elapsed_target','internal_address',
                                  'platform','real_address',
                                  'remote_address','version',
                                  'engine_name','engine_version',
                                  'fast_scale_down','constant_bitrate',
                                  'use_fixed_rates'], axis=1)
    dataset.dropna(inplace=True)
    print('qtd values missing:\n', dataset.isna().sum())
    print('Last shape =>', dataset.shape)
    return dataset

if __name__ == '__main__':
    """
    base_2015, base_2016, base_2017, base_2018 = load_dataset('../arquivos/base_bruta/br2015-2.json', 
             '../arquivos/br2016.json', 
             '../arquivos/br2017full.json',
            '../arquivos/br2018.json')
    """
    dataset_2018 = load_dataset('../base_vazao_2018_2.csv')
    """
    ####### data from year 2015 #########
    dataset_2015 = base_2015.copy()
    print('Sum values missing initial:\n', dataset_2015.isnull().sum())
    dataset_2015 = drop_col_values_missing(dataset_2015)
    dataset_2015 = drop_not_year(dataset_2015, 2015)
    dataset_2015 = cria_sessao(dataset_2015)
    # saving the dataframe 
    dataset_2015.to_csv('../base_vazao_sessao_2015.csv', encoding='utf-8', index=False)
    """

    """
    ####### data from year 2016 #########
    dataset_2016 = base_2016.copy()
    print('Sum values missing initial:\n', dataset_2016.isna().sum())
    dataset_2016 = drop_col_values_missing(dataset_2016)
    dataset_2016 = drop_not_year(dataset_2016, 2016)
    dataset_2016 = cria_sessao(dataset_2016)
    dataset_2016.to_csv('../base_vazao_sessao_2016.csv', encoding='utf-8', index=False)
    """
    """
    ####### data from year 2017 ########
    dataset_2017 = base_2017.copy()
    print('Sum initial missing values:\n', dataset_2017.isna().sum())
    dataset_2017 = drop_col_values_missing_another(dataset_2017)
    dataset_2017 = drop_not_year(dataset_2017, 2017)
    dataset_2017 = cria_sessao(dataset_2017)
    dataset_2017.to_csv('../base_vazao_sessao_2017.csv', encoding='utf-8', index=False)
    """

    ####### data from year 2018 #########
    # dataset_2018 = base_2018.copy()
    print('Sum values missing initial:\n')
    print(dataset_2018.isnull().sum())
    # dataset_2018 = drop_col_values_missing_another(dataset_2018)
    # dataset_2018 = drop_not_year(dataset_2018, 2018)
    dataset_2018 = cria_sessao(dataset_2018)
    dataset_2018.to_csv('../base_vazao_sessao_2018.csv', encoding='utf-8', index=False)
    print('file was successfully created!')