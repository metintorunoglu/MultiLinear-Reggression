# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X=LabelEncoder()
X[:, 3]=le_X.fit_transform(X[:, 3])
enc=OneHotEncoder(categorical_features=[3])
X=enc.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Fit the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)

#Predicting the test set results
y_pred=reg.predict(X_test)

#Building the optimal model by backward elemination (in the second line 
#we're addin b0=1 to all lines (adding constant b0) 
import statsmodels.formula.api as sm 
X=np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt=X[:,[0, 1, 2, 3, 4, 5]]
reg_OLS=sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
#reg_OLS.summary() command reveals the P values for ecah X column
X_opt=X[:,[0, 1, 3, 4, 5]]
reg_OLS=sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt=X[:,[0, 1, 3, 4, 5]]
reg_OLS=sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt=X[:,[0, 3, 4, 5]]
reg_OLS=sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt=X[:,[0, 3, 5]]
reg_OLS=sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()



