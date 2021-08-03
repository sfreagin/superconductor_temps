import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.decomposition import PCA


print("\nHello!\n\nThis Random Forest regression model predicts superconductor critical temperatures.")

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
## Random Forest model
########################################################################

print("\nThe first model uses default parameters")
print(f"\nThe train set has {len(y_train)} samples and the test set has {len(y_test)} samples\n")

#instantiate and fit the model
rfr = RandomForestRegressor()
rfr.fit(X_train_sc,y_train)

#make predictions and measure
y_preds = rfr.predict(X_test_sc)

rmse = np.sqrt(metrics.mean_squared_error(y_test,y_preds))
mae = metrics.mean_absolute_error(y_test,y_preds)
mape = metrics.mean_absolute_percentage_error(y_test,y_preds)
ex_var = metrics.explained_variance_score(y_test,y_preds)
r2_score = metrics.r2_score(y_test,y_preds)


print(f"Root Mean Squared Error:	{np.round(rmse,2)} Kelvins")
print(f"Mean Absolute Error:		{np.round(mae,2)} Kelvins")
print(f"Mean Abs. Percent Error:	{np.round(mape,2)}%")
print(f"Explained Variance Score:	{np.round(ex_var,4)}")
print(f"R2 Score:			{np.round(r2_score,4)}")

########################################################################
## Random Forest model with PCA dimensionality reduction
########################################################################

print("\n****************************************************************")
print("Now we reduce dimensionality using a PCA algorithm to capture 95% of the explained variance\n")

#instantiate PCA with enough dimensions to capture 95% of explained variance
pca = PCA(0.95)
pca.fit(X_train_sc)

X_train_pca = pca.transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

print(f"The number of features is reduced from {X_train_sc.shape[1]} to {X_train_pca.shape[1]}")

rfr = RandomForestRegressor()
rfr.fit(X_train_pca, y_train)


#make predictions
pca_preds = rfr.predict(X_test_pca)

rmse = np.sqrt(metrics.mean_squared_error(y_test,pca_preds))
mae = metrics.mean_absolute_error(y_test,pca_preds)
mape = metrics.mean_absolute_percentage_error(y_test,pca_preds)
ex_var = metrics.explained_variance_score(y_test,pca_preds)
r2_score = metrics.r2_score(y_test,pca_preds)


print("\nUpdated scoring metrics with reduced dimensions:\n")
print(f"Root Mean Squared Error: 	{np.round(rmse,2)} Kelvins")
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
## Random Forest model
########################################################################

print("\nThe first model uses default parameters")
print(f"\nThe train set has {len(y_train)} samples and the test set has {len(y_test)} samples\n")

#instantiate and fit the model
rfr = RandomForestRegressor()
rfr.fit(X_train_sc,y_train)

#make predictions and measure
y_preds = rfr.predict(X_test_sc)

rmse = np.sqrt(metrics.mean_squared_error(y_test,y_preds))
mae = metrics.mean_absolute_error(y_test,y_preds)
mape = metrics.mean_absolute_percentage_error(y_test,y_preds)
ex_var = metrics.explained_variance_score(y_test,y_preds)
r2_score = metrics.r2_score(y_test,y_preds)


print(f"Root Mean Squared Error:	{np.round(rmse,2)} Kelvins")
print(f"Mean Absolute Error:		{np.round(mae,2)} Kelvins")
print(f"Mean Abs. Percent Error:	{np.round(mape,2)}%")
print(f"Explained Variance Score:	{np.round(ex_var,4)}")
print(f"R2 Score:			{np.round(r2_score,4)}")

########################################################################
## Random Forest model with PCA dimensionality reduction
########################################################################

print("\n****************************************************************")
print("Now we reduce dimensionality using a PCA algorithm to capture 95% of the explained variance\n")

#instantiate PCA with enough dimensions to capture 95% of explained variance
pca = PCA(0.95)
pca.fit(X_train_sc)

X_train_pca = pca.transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

print(f"The number of features is reduced from {X_train_sc.shape[1]} to {X_train_pca.shape[1]}")

rfr = RandomForestRegressor()
rfr.fit(X_train_pca, y_train)


#make predictions
pca_preds = rfr.predict(X_test_pca)

rmse = np.sqrt(metrics.mean_squared_error(y_test,pca_preds))
mae = metrics.mean_absolute_error(y_test,pca_preds)
mape = metrics.mean_absolute_percentage_error(y_test,pca_preds)
ex_var = metrics.explained_variance_score(y_test,pca_preds)
r2_score = metrics.r2_score(y_test,pca_preds)


print("\nUpdated scoring metrics with reduced dimensions:\n")
print(f"Root Mean Squared Error: 	{np.round(rmse,2)} Kelvins")
print(f"Mean Absolute Error:		{np.round(mae,2)} Kelvins")
print(f"Mean Abs. Percent Error:	{np.round(mape,2)}%")
print(f"Explained Variance Score:	{np.round(ex_var,4)}")
print(f"R2 Score:			{np.round(r2_score,4)}")