# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:21:19 2023

@author: cianw
"""
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

import matplotlib.pyplot as plt
import copy
import time as time_t
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Packages needed for regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import xgboost

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

#Import of main dataset and creating a target variable, also analysing for outliers of the target variable predecessor 
rds = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")
#Harmonised Risk indicator is based off AVG from 2011-2013, so only want to keep 2013 differences onwards for calculation.
rds = rds[~rds['TIME_PERIOD'].isin([2011, 2012])]
colsToDrop = ['Unnamed: 0', 'subst_cat', 'geo', 'TIME_PERIOD' ]
rds = rds.drop(colsToDrop, axis=1)
colsToDrop2 = list(rds.filter(regex=r'_unit').columns) 
rds = rds.drop(colsToDrop2, axis=1)
rds = rds._get_numeric_data()

Xfull = rds.drop('harmRiskInd', axis=1)
Yfull = rds['harmRiskInd'].values
#global X_train, Y_train, X_val, X_finaltest, Y_val, Y_finaltest
#random_state = 123 chosen for reporducability and because who doesn't love a nice number like that

X_train, X_test, Y_train, Y_test = train_test_split(Xfull, Yfull, test_size=0.6666, random_state=123)



possibleModels  = {
          #'AdaBoostRegressor' : AdaBoostRegressor Extremely unfitting, removed from comparison
          'HuberRegressor': HuberRegressor
          ,'DummyRegressor': DummyRegressor
          ,'TheilSenRegressor' : TheilSenRegressor
          ,'DecisionTree' : DecisionTreeRegressor
          ,'PoissonRegressor': PoissonRegressor
          ,'ElasticNetCV' : ElasticNetCV
          ,'RandomForest': RandomForestRegressor
          ,'LinearRegression': LinearRegression
          ,'XGBoost': XGBRegressor
          ,'ExtraTrees' : ExtraTreesRegressor
          ,'SGD': SGDRegressor
          ,'KN': KNeighborsRegressor
          #,'SVC' : SVC - Considered but took forever to run
          }

def model_test(m):
    print('MODEL START')
    
    name, model_type = m 
    startRun = time_t.time()
    #Apply a scaler to the data using pipeline
    model = Pipeline ([('standardize', MinMaxScaler(feature_range = (0,1))), (name, model_type())])
    #push through pipeline
    mse = cross_val_score(model, X_train, Y_train, cv=KFold(n_splits=2), scoring = 'neg_mean_squared_error', n_jobs=-1)
    r2 = cross_val_score(model, X_train, Y_train, cv=KFold(n_splits=2), scoring = 'r2', n_jobs=-1)
    runtime = int(time_t.time() - startRun)
    
    print('MODEL END')
    return (name, model, r2, np.median(r2), mse, np.median(mse), runtime)

results={}
for name, model_type in possibleModels.items():
    print((name, model_type))
    results[name]=model_test((name, model_type))

resultsOfModelSearch = pd.DataFrame.from_dict(
                                               results
                                              ,orient='index'
                                              ,columns=['name', 'model', 'r2', 'median_r2', 'mse', 'median_mse', 'runtime']
                                             )    


resultsOfModelSearch.to_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\Run01.csv') 
resultsOfModelSearch_save_test = pd.read_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\Run01.csv')
resultsOfModelSearch_save_test.head()

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(10,0), useMathText=True)
ax1 = sns.barplot(x="name"
                  ,y="median_mse"
                  ,data=resultsOfModelSearch 
                  ,palette='mako'
                  ,order=resultsOfModelSearch.sort_values('median_mse', ascending=False).name 
                 )
ax1.set_xticklabels(resultsOfModelSearch.sort_values('median_mse', ascending=False).name,rotation=90)
ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Square Error')
ax1.invert_yaxis()
ax1.set_title("Median Mean Square Error")
ax1.figure.tight_layout()
ax1.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA1\finalRun\medianMSE_2022111.png', dpi= 600) 
















"""
rds['harmRiskInd_next'] = rds.groupby(['geo'])['harmRiskInd'].shift(-1)
rds['increase'] = np.where((rds['harmRiskInd'] <= rds['harmRiskInd_next']), 1, 0)
rds['forAnalysis'] = rds['harmRiskInd_next'].apply(lambda x: 0 if pd.isnull(x) else 1)

sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd", hue="geo")
plt.show()
sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd")
plt.show()
#This is used as justification to remove BG, as it is clear it has outliers such as the value of 240, and it clearly has undue influence on the trend
rds = rds[rds['geo'] != 'BG']


sns.regplot(x='gini', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
sns.regplot(x='productivity', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
sns.regplot(x='waste', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
sns.regplot(x='eduSpend_eur_hab', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
sns.regplot(x='Agricultural sciences', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
sns.regplot(x='pest_KG', y='increase', data =rds , logistic = True,  ci=None)
plt.show()
"""





