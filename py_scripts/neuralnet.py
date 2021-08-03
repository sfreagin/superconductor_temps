import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.decomposition import PCA


print("\nHello!\n\nThis Neural Net regression model predicts superconductor critical temperatures.")

########################################################################
## Setting up the data
########################################################################

#pull in DataFrame
superconductor_df = pd.read_csv('../datasets/train.csv')

#set up X and y
X = superconductor_df.drop(columns=['critical_temp'])
y = superconductor_df['critical_temp']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#instantiate scaler and scale data
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

########################################################################
## Neural Net model
########################################################################

print("\nThe first model uses default parameters")
print(f"\nThe train set has {len(y_train)} samples and the test set has {len(y_test)} samples\n")

#set up the model, compile, and fit
model = Sequential()

model.add(Dense(32, input_dim = X_train_sc.shape[1], activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

print("********First model summary********")
print(model.summary())

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae'])
history = model.fit(X_train_sc, y_train, epochs = 150, batch_size = 16, verbose = 0, validation_split = 0.2)

#make predictions
y_preds = model.predict(X_test_sc)

rmse = np.sqrt(metrics.mean_squared_error(y_test,y_preds))
mae = metrics.mean_absolute_error(y_test,y_preds)
mape = metrics.mean_absolute_percentage_error(y_test,y_preds)
ex_var = metrics.explained_variance_score(y_test,y_preds)
r2_score = metrics.r2_score(y_test,y_preds)

print("\n************First model results************\n")
print(f"Root Mean Squared Error:	{np.round(rmse,2)} Kelvins")
print(f"Mean Absolute Error:		{np.round(mae,2)} Kelvins")
print(f"Mean Abs. Percent Error:	{np.round(mape,2)}%")
print(f"Explained Variance Score:	{np.round(ex_var,4)}")
print(f"R2 Score:			{np.round(r2_score,4)}")


########################################################################
## Doing it all again with the other DataFrame
########################################################################

print("\n****************************************************************")
print("\n****************************************************************")
print("Now let's run the same analysis on the other DataFrame")
print("\n****************************************************************")


elements_df = pd.read_csv('../datasets/unique_m.csv')

X = elements_df.drop(columns=['critical_temp','material'])
y = elements_df['critical_temp']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#instantiate scaler and scale data
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

########################################################################
## Neural Net model
########################################################################

print("\nThe first model uses default parameters")
print(f"\nThe train set has {len(y_train)} samples and the test set has {len(y_test)} samples\n")

#set up the model, compile, and fit
model = Sequential()

model.add(Dense(32, input_dim = X_train_sc.shape[1], activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

print("********Second model summary********")
print(model.summary())

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse', 'mae'])
history = model.fit(X_train_sc, y_train, epochs = 150, batch_size = 16, verbose = 0, validation_split = 0.2)

#make predictions
y_preds = model.predict(X_test_sc)

rmse = np.sqrt(metrics.mean_squared_error(y_test,y_preds))
mae = metrics.mean_absolute_error(y_test,y_preds)
mape = metrics.mean_absolute_percentage_error(y_test,y_preds)
ex_var = metrics.explained_variance_score(y_test,y_preds)
r2_score = metrics.r2_score(y_test,y_preds)

print("\n************Second model results************\n")
print(f"Root Mean Squared Error:	{np.round(rmse,2)} Kelvins")
print(f"Mean Absolute Error:		{np.round(mae,2)} Kelvins")
print(f"Mean Abs. Percent Error:	{np.round(mape,2)}%")
print(f"Explained Variance Score:	{np.round(ex_var,4)}")
print(f"R2 Score:			{np.round(r2_score,4)}")
