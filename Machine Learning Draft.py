# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:21:19 2023

@author: cianw
"""
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

import scipy
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import lognorm
from scipy.stats import ranksums
from scipy.stats import f_oneway
from scipy.stats import tmean
from scipy.stats import tstd

#Import of main dataset and creating a target variable, also analysing for outliers of the target variable predecessor 
rds = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")
#Harmonised Risk indicator is based off AVG from 2011-2013, so only want to keep 2013 differences onwards for calculation.
#rds = rds[~rds['TIME_PERIOD'].isin([2011, 2012])]

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






