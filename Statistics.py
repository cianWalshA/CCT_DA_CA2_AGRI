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
from scipy.stats import ranksums
from scipy.stats import f_oneway

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

sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd", hue="geo")
plt.show()
sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd")
plt.show()


"""
This section will compare populations, such as early Harmonised Risk vs later, early Median incomes, male median incomes vs female, 
"""
"""
Change in Risk
"""
hri = rds[['TIME_PERIOD', 'geo', 'harmRiskInd']]
hri_14_15 = hri[hri['TIME_PERIOD'].isin([2014, 2015, 2016])]
hri_18_19 = hri[hri['TIME_PERIOD'].isin([2018, 2019, 2020])]

print(shapiro(hri_14_15['harmRiskInd']))
print(kstest(hri_14_15['harmRiskInd'], 'norm'))

print(shapiro(hri_18_19['harmRiskInd']))
print(kstest(hri_18_19['harmRiskInd'], 'norm'))

#It is clear these are all non normal distributions, lets plot to investigate similarities

sns.histplot(data = hri_14_15 , x = 'harmRiskInd', color="skyblue", label="previous", kde=True)
sns.histplot(data = hri_18_19 , x= 'harmRiskInd', color="red", label="Current" , kde=True)
plt.show()

#It definitely appears that the risk levels have in general shifted down, with some exceptions.
#As the data is shown as non normal using Shapiro and KSTEST, will compare using nonparametric tests.

ranksums(hri_14_15['harmRiskInd'], hri_18_19['harmRiskInd'])
f_oneway(hri_14_15['harmRiskInd'], hri_18_19['harmRiskInd'])

#Testing for normality shows failure, so will use non parametric tests 
"""
MEDIAN INCOME TEST
"""
medIncome = rds[['TIME_PERIOD', 'geo', 'MED_EF', 'MED_EM']]
medIncome_M = rds[['TIME_PERIOD', 'geo', 'MED_EM']].dropna()
medIncome_F = rds[['TIME_PERIOD', 'geo', 'MED_EF']].dropna()

print(shapiro(medIncome_M['MED_EM']))
print(kstest(medIncome_M['MED_EM'], 'norm'))

print(shapiro(medIncome_F['MED_EF']))
print(kstest(medIncome_F['MED_EF'], 'norm'))


#It is clear these are all non normal distributions, lets plot to investigate similarities

sns.histplot(data = medIncome_M , x = 'MED_EM', color="skyblue", label="previous", kde=True)
sns.histplot(data = medIncome_F , x= 'MED_EF', color="red", label="Current" , kde=True)
plt.show()

#It definitely appears that the risk levels have in general shifted down, with some exceptions.
#As the data is shown as non normal using Shapiro and KSTEST, will compare using nonparametric tests.

ranksums(medIncome_M['MED_EM'], medIncome_F['MED_EF'])
f_oneway(medIncome_M['MED_EM'], medIncome_F['MED_EF'])
#CAN ASESS USING THIS AND THE HISTPLOT THAT THE POPULATIONS ARE LIKELY SIMILAR. AS P VALUE IS GREATER THAN 0.05 @0.325 































