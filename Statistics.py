# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import lognorm

rds = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")

harmRiskInd = rds[rds.columns[rds.columns.isin(['TIME_PERIOD', 'geo', 'harmRiskInd'])]]
harmRiskInd['harmRiskInd_next'] = harmRiskInd.groupby(['geo'])['harmRiskInd'].shift(-1)
harmRiskInd['increase'] = np.where((harmRiskInd['harmRiskInd'] <= harmRiskInd['harmRiskInd_next']), 1, 0)
harmRiskInd['forAnalysis'] = harmRiskInd['harmRiskInd_next'].apply(lambda x: 0 if pd.isnull(x) else 1)

hri_Other = harmRiskInd[harmRiskInd['geo'] != 'IE']
hri_Other_summary = hri_Other.groupby(['TIME_PERIOD'])['harmRiskInd'].agg(['mean', 'median', 'count', 'std'])

hri2012 = harmRiskInd[harmRiskInd['TIME_PERIOD'] == 2012]
print(shapiro(hri2012['harmRiskInd']))
sns.histplot(data = hri2012, x='harmRiskInd', bins=35)
plt.show()
hri_Ire = harmRiskInd[harmRiskInd['geo'] == 'IE']


sns.histplot(data = harmRiskInd, x='harmRiskInd', bins=35)
plt.show()
sns.lineplot(data=harmRiskInd, x="TIME_PERIOD", y="harmRiskInd", hue="geo")
plt.show()
sns.lineplot(data=harmRiskInd, x="TIME_PERIOD", y="harmRiskInd")
plt.show()
#This is used as justification to remove BG, as it is clear it has outliers such as the value of 240, and it clearly has undue influence on the trend

