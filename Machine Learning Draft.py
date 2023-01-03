# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:21:19 2023

@author: cianw
"""
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

import matlotlib
import matplotlib.pyplot as plt
import copy
import time as time_t
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from xgboost import XGBRegressor
import xgboost

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

#Import of main dataset and creating a target variable, also analysing for outliers of the target variable predecessor 
rds = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")

colsToDrop2 = list(rds.filter(regex=r'_unit|year|Unnamed').columns) 
rds = rds.drop(colsToDrop2, axis=1)
rds = rds._get_numeric_data()

scaler = MinMaxScaler(feature_range = (0,1), copy = False).fit(rds) 
rds_scaler = scaler.transform(rds)
rds_scaler = pd.DataFrame(rds_scaler, columns = rds.columns)

colsToDropHICP= list(rds.filter(regex=r'hicp').columns)
Xfull = rds.drop(colsToDropHICP, axis=1)
Yfull = rds['hicp_n12m'].values


X_train, X_temp, Y_train, Y_temp = train_test_split(Xfull, Yfull, test_size=0.6666, random_state=123)
X_val, X_finaltest, Y_val, Y_finaltest = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=123)

#drop temp data used for splitting test w/ validation
del X_temp, Y_temp

"""
I have created an absurd amount of features:
    Going to cut them down significantly in order to make use of them effectively, decide on 2 models without overfitting. 
    
"""

#Backward Elimination
cols = list(X_train.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X_train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
    
selected_features_BE = cols

#Use Lasso to pick features
reg = LassoCV()
reg.fit(X_train[selected_features_BE], Y_train)
#print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
#print("Best score using built-in LassoCV: %f" % reg.score(X_train[selected_features_BE],Y_train))
coef = pd.Series(reg.coef_, index = X_train[selected_features_BE].columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

step2Columns = pd.DataFrame(coef, columns=['score']).reset_index()

columnsForModel = step2Columns[step2Columns['score']!=0]

"""
Begin Machine Learning model selection process
"""

X_train, X_temp, Y_train, Y_temp = train_test_split(Xfull[columnsForModel['index']], Yfull, test_size=0.6666, random_state=123)
X_val, X_finaltest, Y_val, Y_finaltest = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=123)

possibleModels  = {
#          'AdaBoostRegressor' : AdaBoostRegressor Extremely unfitting, removed from comparison
#          'HuberRegressor': HuberRegressor
#          ,'DummyRegressor': DummyRegressor
#          ,'TheilSenRegressor' : TheilSenRegressor
          'DecisionTree' : DecisionTreeRegressor
#          ,'PoissonRegressor': PoissonRegressor
#          ,'ElasticNetCV' : ElasticNetCV
          ,'RandomForest': RandomForestRegressor
          ,'LinearRegression': LinearRegression
          ,'XGBoost': XGBRegressor
          ,'ExtraTrees' : ExtraTreesRegressor
#          ,'SGD': SGDRegressor
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
    mse = cross_val_score(model, X_train, Y_train, cv=KFold(n_splits=3), scoring = 'neg_mean_squared_error', n_jobs=-1)
    r2 = cross_val_score(model, X_train, Y_train, cv=KFold(n_splits=3), scoring = 'r2', n_jobs=-1)
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
resultsOfModelSearch = resultsOfModelSearch[resultsOfModelSearch['median_r2']>0]

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
ax1.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\medianMSE_20230102.png', dpi= 600) 

fig, ax2 = plt.subplots(figsize=(10,6))
ax2 = sns.barplot(x="name"
                  ,y="median_r2"
                  ,data=resultsOfModelSearch 
                  ,palette='mako'
                  ,order=resultsOfModelSearch.sort_values('median_r2', ascending=False).name 
                 )
ax2.set_xticklabels(resultsOfModelSearch.sort_values('median_r2', ascending=False).name,rotation=90)
ax2.set_xlabel('Model')
ax2.set_ylabel('R-Squared')
ax2.set_title("Median R-Squared")
ax2.figure.tight_layout()
ax2.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\medianR2_20230102.png', dpi= 600) 

#Test Models = ExtraTrees, XGBoost, RandomForest

hyperparameters_xgb = {'xgbregressor__max_depth': range(3, 11, 2)
                       ,'xgbregressor__n_estimators' : range(50, 300, 50)
                       ,'xgbregressor__learning_rate' : [ 0.01, 0.05, 0.1, 0.15, 0.2]}
hyperparameters_xtra = {'extratreesregressor__min_samples_split': range(2, 10, 2)
                       ,'extratreesregressor__n_estimators' : range(50, 300, 50)
                       ,'extratreesregressor__min_samples_leaf' :  range(2, 10, 2) }
hyperparameters_rf = {'randomforestregressor__min_samples_split': range(2, 10, 2)
                      ,'randomforestregressor__n_estimators' : range(50, 300, 50)
                       ,'randomforestregressor__min_samples_leaf' :  range(2, 10, 2) }

# Set up the pipeline containing the scalers
#pipeline_knn = make_pipeline(MinMaxScaler(feature_range = (0,1)), 
#                         KNeighborsRegressor())
pipeline_xgb = make_pipeline(MinMaxScaler(feature_range = (0,1)),
                         xgboost.XGBRegressor())
pipeline_xtra = make_pipeline(MinMaxScaler(feature_range = (0,1)),
                         ExtraTreesRegressor())
pipeline_rf = make_pipeline(MinMaxScaler(feature_range = (0,1)),
                         RandomForestRegressor())

final_results = {}
for model_values in [
                    #(pipeline_knn,  hyperparameters_knn,  'KN')
                    (pipeline_xgb, hyperparameters_xgb, 'XGBoost')
                    ,(pipeline_xtra, hyperparameters_xtra, 'ExtraTrees')
                    ,(pipeline_rf, hyperparameters_rf, 'RandomForest')
                     ]:
    print('GridSearch Start: ' + str(model_values[2]))
    clf = GridSearchCV(model_values[0], model_values[1], cv = 10, scoring  = 'r2', n_jobs =-1)
    clf.fit(X_val, Y_val)
    name = model_values[2]
    final_results[name] = clf
    
    print ("Hyperparameter results for {}".format(name))
    print ("\tBest Score: {}".format(clf.best_score_))
    print ("\tBest params: {}".format(clf.best_params_))
    print('GridSearch Finish: ' + str(model_values))


XGB_gridsearchcv_summary = pd.DataFrame(final_results['XGBoost'].cv_results_)
XGB_gridsearchcv_summary['name'] = 'XGB'
XTRA_gridsearchcv_summary = pd.DataFrame(final_results['ExtraTrees'].cv_results_)
XTRA_gridsearchcv_summary['name'] = 'XTRA'
RF_gridsearchcv_summary = pd.DataFrame(final_results['RandomForest'].cv_results_)
RF_gridsearchcv_summary['name'] = 'RF'
#Save these because who trusts temporary memory??
XGB_gridsearchcv_summary.to_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\XGB_gridsearchcv_summary_w_dist_20230103.csv') 
XTRA_gridsearchcv_summary.to_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\XTRA_gridsearchcv_summary_w_dist_20230103.csv')
RF_gridsearchcv_summary.to_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\RF_gridsearchcv_summary_w_dist_20230103.csv')


bestValues = pd.concat([XGB_gridsearchcv_summary[XGB_gridsearchcv_summary['rank_test_score'].isin([1, 2, 3])]
                    ,XTRA_gridsearchcv_summary[XTRA_gridsearchcv_summary['rank_test_score'].isin([1, 2, 3])]
                    ,RF_gridsearchcv_summary[RF_gridsearchcv_summary['rank_test_score'].isin([1, 2, 3])]]
                    ,join='inner')
bestValues.sort_values('rank_test_score', ascending=True).head(9)

bestValues.sort_values('rank_test_score', ascending=True).head(9)
bestValues.to_csv(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\gridsearch_top9_w_dist_20230103.csv')

bestValues['OutputLabels'] = bestValues['name'] + ' ' + bestValues['rank_test_score'].astype(str)
fig = plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x="OutputLabels"
                  ,y="mean_test_score"
                  ,data=bestValues 
                  ,palette="mako"
                  #,hue="rank_test_score"
                  ,order=bestValues.sort_values('mean_test_score', ascending=False).OutputLabels 
                 )
ax3.set_xticklabels(bestValues.sort_values('mean_test_score', ascending=False).OutputLabels,rotation=90)
ax3.set_xlabel('Hyperparameters')
ax3.set_ylabel('Mean Test Score')
ax3.set_title("Highest Mean Test Scores for Hyperparameter Combinations")
ax3.figure.tight_layout()
ax3.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230102\best_score__20230103.png', dpi= 600) 

#Create a final Train 
X = np.vstack((X_train, X_val))
Y = np.hstack((Y_train, Y_val))
scaler = MinMaxScaler(feature_range = (0,1), copy = False).fit(X) 

X = scaler.transform(X)
X_columns = X_train.columns
X_final = pd.DataFrame(X, columns = X_columns)

X_finaltest = scaler.transform(X_finaltest)
X_finaltest = pd.DataFrame(X_finaltest, columns = X_columns)

# Fit an XGBRegressor model to the training data. 
# Use tuned hyperparameters here 
RF_model  = ExtraTreesRegressor(min_samples_leaf = 2, min_samples_split = 2, n_estimators = 50)
RF_model  = RF_model.fit(X,Y)
# Use the fitted model to predict the values from the test data
Y_pred = RF_model .predict(X_finaltest)
# Evaluate the performance of the model at correctly predicting the values of the test data
m_mse = mean_squared_error(Y_finaltest, Y_pred)
m_r2 = r2_score(Y_finaltest, Y_pred)




