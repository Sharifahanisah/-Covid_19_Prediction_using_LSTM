# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:33:43 2022

@author: HP
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Dense,Dropout,LSTM,Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.keras import Input

#%% constant
CSV_PATH_TRAIN = os.path.join(os.getcwd(),'dataset',
                              'cases_malaysia_train.csv')

CSV_PATH_TEST = os.path.join(os.getcwd(),'dataset',
                              'cases_malaysia_test.csv')

MMS_PATH = os.path.join(os.getcwd(),'Models','mms.pkl')

LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH_TRAIN, na_values=(" ","?"))
df_test = pd.read_csv(CSV_PATH_TEST)


#%% Step 2) Data Inspection
df.head()
df.tail()
df.info()
df.isna().sum()
df.describe().T

df_disp = df[100:200]

plt.figure()
plt.plot(df_disp['cases_new'])
plt.show()

df_test.head()
df_test.tail()
df_test.info()
df_test.isna().sum()
df_test.describe().T

df_test_disp = df[100:200]

plt.figure()
plt.plot(df_test_disp['cases_new'])
plt.show()



#%% Step 3) Data Cleaning 

df = df.drop(labels = ['cluster_import','cluster_religious',
                      'cluster_community','cluster_highRisk',
                      'cluster_education','cluster_detentionCentre',
                      'cluster_workplace'], axis=1)

df_test = df_test.drop(labels = ['cluster_import','cluster_religious',
                      'cluster_community','cluster_highRisk',
                      'cluster_education','cluster_detentionCentre',
                      'cluster_workplace'], axis=1)

df = df.interpolate(method='polynomial', order=2)
df['cases_new'] = df['cases_new'].apply(np.floor) 

df_test = df_test.interpolate(method='polynomial', order=2)
df_test['cases_new'] = df_test['cases_new'].apply(np.floor) 

#%% Step 4) Features Selection

# Train dataset
X = df['cases_new'] # only 1 feature

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms,file)

win_size = 30
X_train = []
y_train = []

for i in range(win_size, len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Test dataset

dataset_cat = pd.concat((df['cases_new'],df_test['cases_new']))

length_days = len(dataset_cat) - len(df_test) - win_size
tot_input = dataset_cat[length_days:]

Xtest = mms.transform(np.expand_dims(tot_input,axis=-1))

X_test = []
y_test = []

for i in range(win_size, len(Xtest)):
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])
    
X_test = np.array(X_test)
y_test = np.array(y_test)

#%% Step 5) Data Preprocessing

#%% Model Development

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Bidirectional(LSTM(64,return_sequences=(True))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(1,activation= 'linear'))
model.summary()

plot_model(model,show_shapes=True,show_layer_names= True)

#%% 
model.compile(optimizer = 'adam', loss='mse',
              metrics = ['mean_absolute_percentage_error','mse'] )
# callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

#%% model training

hist= model.fit(X_train,y_train,
                epochs= 350, 
                callbacks = [tensorboard_callback],
                validation_data=(X_test,y_test))

#%% model evaluation
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.legend(['Training MSE','Validation MSE'])
plt.show()

#%%
predicted_Number_of_cases = model.predict(X_test)

plt.figure()
plt.plot(y_test, color ='red')
plt.plot(predicted_Number_of_cases, color ='blue')
plt.xlabel('Time')
plt.ylabel('Number of cases')
plt.legend(['Actual','Predicted'])
plt.show()

actual_cases = mms.inverse_transform(y_test)
predicted_cases = mms.inverse_transform(predicted_Number_of_cases)

plt.figure()
plt.plot(actual_cases , color ='red')
plt.plot(predicted_cases, color ='blue')
plt.xlabel('Time')
plt.ylabel('Number of cases')
plt.legend(['Actual','Predicted'])
plt.show()

#%%
print(mean_absolute_error(actual_cases,predicted_cases))
print(mean_squared_error(actual_cases,predicted_cases))
print(mean_absolute_percentage_error(actual_cases,predicted_cases)*100)
