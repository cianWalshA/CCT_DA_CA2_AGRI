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

sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd", hue="geo")
plt.show()
sns.lineplot(data=rds, x="TIME_PERIOD", y="harmRiskInd")
plt.show()


"""
STATISTICS QUESTION 2
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

sns.lineplot(data = medIncome_M , x='TIME_PERIOD',  y= 'MED_EM', color="skyblue", label="Median Male Income")
sns.lineplot(data = medIncome_F , x='TIME_PERIOD',  y= 'MED_EF', color="red", label="Median Female Income")
plt.show()
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


"""
GINI WELLBEING IMPROVEMENT TEST
"""
gini = rds[['TIME_PERIOD', 'geo', 'gini']]
gini_11_13 = gini[gini['TIME_PERIOD'].isin([2011, 2012, 2013])].dropna()
gini_18_20 = gini[gini['TIME_PERIOD'].isin([2018, 2019, 2020])].dropna()


sns.lineplot(data = gini , x='TIME_PERIOD',  y= 'gini', color="skyblue", label="AVG GINI")
#sns.lineplot(data = gini_18_20 , x='TIME_PERIOD',  y= 'gini', color="red", label="AVG GINI 18-20")
plt.show()

print(shapiro(gini_11_13['gini']))
print(kstest(gini_11_13['gini'], 'norm'))

print(shapiro(gini_18_20['gini']))
print(kstest(gini_18_20['gini'], 'norm'))

print(gini_11_13['gini'].describe())
print(gini_18_20['gini'].describe())

#It is clear these are all non normal distributions, lets plot to investigate similarities

sns.histplot(data = gini_11_13 , x = 'gini', color="skyblue", label="previous", kde=True)
sns.histplot(data = gini_18_20 , x= 'gini', color="red", label="Current" , kde=True)
plt.show()

#It definitely appears that the risk levels have in general shifted down, with some exceptions.
#As the data is shown as non normal using Shapiro and KSTEST, will compare using nonparametric tests.
gini_11_13['gini']
ranksums(gini_11_13['gini'], gini_18_20['gini'])
f_oneway(gini_11_13['gini'], gini_18_20['gini'])
#CAN ASESS USING THIS AND THE HISTPLOT THAT THE POPULATIONS ARE LIKELY SIMILAR. AS P VALUE IS GREATER THAN 0.05 @0.325 

"""
Improvments in Biodiversity of birds:
    Mao Exterminated all the birds, gave way to plague of locusts.... 
"""
bird = rds[['TIME_PERIOD', 'geo', 'birdBiodiversityIndex']]
bird_11_13 = bird[bird['TIME_PERIOD'].isin([2011, 2012])].dropna()
bird_18_20 = bird[bird['TIME_PERIOD'].isin([2018, 2019])].dropna()

sns.lineplot(data = bird , x='TIME_PERIOD',  y= 'birdBiodiversityIndex', color="skyblue", label="AVG Biodiversity")
plt.show()


print(shapiro(bird_11_13['birdBiodiversityIndex']))
print(kstest(bird_11_13['birdBiodiversityIndex'], 'norm'))

print(shapiro(bird_18_20['birdBiodiversityIndex']))
print(kstest(bird_18_20['birdBiodiversityIndex'], 'norm'))

print(bird_11_13['birdBiodiversityIndex'].describe())
print(bird_18_20['birdBiodiversityIndex'].describe())

#It is clear these are all non normal distributions, lets plot to investigate similarities

sns.histplot(data = bird_11_13 , x = 'birdBiodiversityIndex', color="skyblue", label="previous", kde=True)
sns.histplot(data = bird_18_20 , x= 'birdBiodiversityIndex', color="red", label="Current" , kde=True)
plt.show()

#It definitely appears that the risk levels have in general shifted down, with some exceptions.
#As the data is shown as non normal using Shapiro and KSTEST, will compare using nonparametric tests.
gini_11_13['birdBiodiversityIndex']
ranksums(bird_11_13['birdBiodiversityIndex'], bird_18_20['birdBiodiversityIndex'])
f_oneway(bird_11_13['birdBiodiversityIndex'], bird_18_20['birdBiodiversityIndex'])
#CAN ASESS USING THIS AND THE HISTPLOT THAT THE POPULATIONS ARE LIKELY SIMILAR. AS P VALUE IS GREATER THAN 0.05 @0.325 




"""
STATISTICS QUESTION 3
This section will compare Ireland in terms of research spending on agriculture when compared to 
"""
rdsIE = rds[rds['geo'] == 'IE']
# Select only the rows for the specified country

agriSpending = rds[['TIME_PERIOD', 'geo', 'Agricultural sciences']]
agriSpending_IE = agriSpending[agriSpending['geo'] == 'IE'].dropna()
agriSpending_EU = agriSpending[agriSpending['geo'] != 'IE'].dropna()

agriSpendingCompare = agriSpending_EU.merge(agriSpending_IE, on='TIME_PERIOD', how='inner', suffixes=('_EU', '_IE'))

agriSpendingCompare['difference'] = 

plt.show()























