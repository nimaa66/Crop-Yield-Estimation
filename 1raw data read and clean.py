# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:39:56 2022

@author: mohammadigolafshani
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import datetime  as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy

NorthShapefile_6 = pd.read_csv('D:/MYDATA/NorthShapefile_6.csv', sep=',')
NorthShapefile_6.isna().sum()

NorthShapefile = pd.read_csv('D:/MYDATA/NorthShapefile.csv', sep=',')
len(pd.unique(NorthShapefile['NUTS_CODE']))
NorthShapefile.isna().sum()

###############################################################################

dataET_1 = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_1.csv', sep=',')
print (len(dataET_1.columns))
dataET_1.isna().sum()

dataET_2 = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_2.csv', sep=',')
print (len(dataET_2.columns))
dataET_2.isna().sum()

dataET_3 = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_3.csv', sep=',')
print (len(dataET_3.columns))
dataET_3.isna().sum()

dataET_4 = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_4.csv', sep=',')
print (len(dataET_4.columns))
dataET_4.isna().sum()

dataET_5 = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_5.csv', sep=',')
print (len(dataET_5.columns))
dataET_5.isna().sum()

dataET_6_ = pd.read_csv('D:/MYDATA/3mahe/dataET/dataET_6.csv', sep=',')
dataET_6_.isna().sum()
dataET_6 = dataET_6_.merge(NorthShapefile_6, on= ['NUTS_CODE', 'crop', 'year' ])
dataET_6.drop(['NUTS_NAME_y', 'FedState_y'], axis=1, inplace=True)
dataET_6.rename(columns = {'FedState_x':'FedState', 'NUTS_NAME_x': 'NUTS_NAME'}, inplace = True)
print (dataET_6.isna().sum())
print (len(dataET_6.columns))

dataET = dataET_1.append([dataET_2, dataET_3, dataET_4, dataET_5, dataET_6], ignore_index = True)
len(pd.unique(dataET['NUTS_CODE']))
dataET.isna().sum()

dataET['date'] =pd.to_datetime(dataET['date'])
dataET['year'] =pd.to_datetime(dataET['year'], format='%Y')
print (dataET.dtypes)

dataET_yeartoyear = dataET.loc[dataET['date'].dt.year == dataET['year'].dt.year]
dataET_yeartoyear.reset_index(drop=True, inplace=True)
dataET_yeartoyear
dataET_yeartoyear.isna().sum()

dataET_yeartoyear.to_csv('D:/dataET_yeartoyear.csv', encoding='utf-8')

###############################################################################

dataNDVI_1 = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_1.csv', sep=',')
print (len(dataNDVI_1.columns))
dataNDVI_1.isna().sum()

dataNDVI_2 = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_2.csv', sep=',')
print (len(dataNDVI_2.columns))
dataNDVI_2.isna().sum()

dataNDVI_3 = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_3.csv', sep=',')
print (len(dataNDVI_3.columns))
dataNDVI_3.isna().sum()

dataNDVI_4 = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_4.csv', sep=',')
print (len(dataNDVI_4.columns))
dataNDVI_4.isna().sum()

dataNDVI_5 = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_5.csv', sep=',')
print (len(dataNDVI_5.columns))
dataNDVI_5.isna().sum()

dataNDVI_6_ = pd.read_csv('D:/MYDATA/3mahe/dataNDVI/dataNDVI_6.csv', sep=',')
dataNDVI_6_.isna().sum()
dataNDVI_6 = dataNDVI_6_.merge(NorthShapefile_6, on= ['NUTS_CODE', 'crop', 'year' ])
dataNDVI_6.drop(['NUTS_NAME_y', 'FedState_y'], axis=1, inplace=True)
dataNDVI_6.rename(columns = {'FedState_x':'FedState', 'NUTS_NAME_x': 'NUTS_NAME'}, inplace = True)
print (dataNDVI_6.isna().sum())
print (len(dataNDVI_6.columns))

dataNDVI = dataNDVI_1.append([dataNDVI_2, dataNDVI_3, dataNDVI_4, dataNDVI_5, dataNDVI_6], ignore_index = True)
len(pd.unique(dataNDVI['NUTS_CODE']))
dataNDVI.isna().sum()

dataNDVI['date'] =pd.to_datetime(dataNDVI['date'])
dataNDVI['year'] =pd.to_datetime(dataNDVI['year'], format='%Y')
print (dataNDVI.dtypes)

dataNDVI_yeartoyear = dataNDVI.loc[dataNDVI['date'].dt.year == dataNDVI['year'].dt.year]
dataNDVI_yeartoyear.reset_index(drop=True, inplace=True)
dataNDVI_yeartoyear
dataNDVI_yeartoyear.isna().sum()

dataNDVI_yeartoyear.to_csv('D:/dataNDVI_yeartoyear.csv', encoding='utf-8')

###############################################################################

dataNDWI_1 = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_1.csv', sep=',')
print (len(dataNDWI_1.columns))
dataNDWI_1.isna().sum()

dataNDWI_2 = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_2.csv', sep=',')
print (len(dataNDWI_2.columns))
dataNDWI_2.isna().sum()

dataNDWI_3 = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_3.csv', sep=',')
print (len(dataNDWI_3.columns))
dataNDWI_3.isna().sum()

dataNDWI_4 = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_4.csv', sep=',')
print (len(dataNDWI_4.columns))
dataNDWI_4.isna().sum()

dataNDWI_5 = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_5.csv', sep=',')
print (len(dataNDWI_5.columns))
dataNDWI_5.isna().sum()

dataNDWI_6_ = pd.read_csv('D:/MYDATA/3mahe/dataNDWI/dataNDWI_6.csv', sep=',')
dataNDWI_6_.isna().sum()
dataNDWI_6 = dataNDWI_6_.merge(NorthShapefile_6, on= ['NUTS_CODE', 'crop', 'year' ])
dataNDWI_6.drop(['NUTS_NAME_y', 'FedState_y'], axis=1, inplace=True)
dataNDWI_6.rename(columns = {'FedState_x':'FedState', 'NUTS_NAME_x': 'NUTS_NAME'}, inplace = True)
print (dataNDWI_6.isna().sum())
print (len(dataNDWI_6.columns))

dataNDWI = dataNDWI_1.append([dataNDWI_2, dataNDWI_3, dataNDWI_4, dataNDWI_5, dataNDWI_6], ignore_index = True)
len(pd.unique(dataNDWI['NUTS_CODE']))
dataNDWI.isna().sum()

dataNDWI['date'] =pd.to_datetime(dataNDWI['date'])
dataNDWI['year'] =pd.to_datetime(dataNDWI['year'], format='%Y')
print (dataNDWI.dtypes)

dataNDWI_yeartoyear = dataNDWI.loc[dataNDWI['date'].dt.year == dataNDWI['year'].dt.year]
dataNDWI_yeartoyear.reset_index(drop=True, inplace=True)
dataNDWI_yeartoyear
dataNDWI_yeartoyear.isna().sum()

dataNDWI_yeartoyear.to_csv('D:/dataNDWI_yeartoyear.csv', encoding='utf-8')

###############################################################################

dataLST_Day_1 = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_1.csv', sep=',')
print (len(dataLST_Day_1.columns))
dataLST_Day_1.isna().sum()

dataLST_Day_2 = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_2.csv', sep=',')
print (len(dataLST_Day_2.columns))
dataLST_Day_2.isna().sum()

dataLST_Day_3 = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_3.csv', sep=',')
print (len(dataLST_Day_3.columns))
dataLST_Day_3.isna().sum()

dataLST_Day_4 = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_4.csv', sep=',')
print (len(dataLST_Day_4.columns))
dataLST_Day_4.isna().sum()

dataLST_Day_5 = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_5.csv', sep=',')
print (len(dataLST_Day_5.columns))
dataLST_Day_5.isna().sum()

dataLST_Day_6_ = pd.read_csv('D:/MYDATA/3mahe/dataLST_Day/dataLST_Day_6.csv', sep=',')
dataLST_Day_6_.isna().sum()
dataLST_Day_6 = dataLST_Day_6_.merge(NorthShapefile_6, on= ['NUTS_CODE', 'crop', 'year' ])
dataLST_Day_6.drop(['NUTS_NAME_y', 'FedState_y'], axis=1, inplace=True)
dataLST_Day_6.rename(columns = {'FedState_x':'FedState', 'NUTS_NAME_x': 'NUTS_NAME'}, inplace = True)
print (dataLST_Day_6.isna().sum())
print (len(dataLST_Day_6.columns))

dataLST_Day = dataLST_Day_1.append([dataLST_Day_2, dataLST_Day_3, dataLST_Day_4, dataLST_Day_5, dataLST_Day_6], ignore_index = True)
len(pd.unique(dataLST_Day['NUTS_CODE']))
dataLST_Day.isna().sum()

dataLST_Day['date'] =pd.to_datetime(dataLST_Day['date'])
dataLST_Day['year'] =pd.to_datetime(dataLST_Day['year'], format='%Y')
print (dataLST_Day.dtypes)

dataLST_Day_yeartoyear = dataLST_Day.loc[dataLST_Day['date'].dt.year == dataLST_Day['year'].dt.year]
dataLST_Day_yeartoyear.reset_index(drop=True, inplace=True)
dataLST_Day_yeartoyear
dataLST_Day_yeartoyear.isna().sum()

dataLST_Day_yeartoyear.to_csv('D:/dataLST_Day_yeartoyear.csv', encoding='utf-8')





















