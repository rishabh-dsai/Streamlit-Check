# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:32:17 2022

@author: asus
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
features = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


#%%

X=pd.get_dummies(features,prefix='State',columns=['State'])
X=X[['R&D Spend', 'Administration', 'Marketing Spend', 'State_California',
       'State_Florida', 'State_New York']]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = pd.Series(regressor.predict(X_test))

#%%

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,y_pred)


df=y_test.reset_index()
df['Predicted']=y_pred
df.drop(columns='index',inplace=True)

df.plot()

#%%

joblib.dump(regressor,'model.sav')

#%%

loaded_model = joblib.load('model.sav')
y_preds=loaded_model.predict(X)


plt.plot(y_preds)
plt.plot(y)
plt.show()




