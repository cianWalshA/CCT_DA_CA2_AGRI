# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
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
#rds = rds[~rds['TIME_PERIOD'].isin([2011, 2012])]=
rds['date'] = pd.to_datetime(rds['date'])

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_title("Trend in HICP by Country")
sns.lineplot(data=rds, x="date", y="hicp_ROC", hue='geo')
ax1.set_xlabel('Month')
ax1.set_ylabel('hicp_ROC')
ax1.figure.tight_layout()
ax1.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230103\Stats\HICP_TREND.png', dpi= 600)
plt.show()

fig, ax2 = plt.subplots(figsize=(10,6))
ax2.set_title("Trend in HICP")
sns.lineplot(data=rds, x="date", y="hicp_ROC")
ax2.set_xlabel('Month')
ax2.set_ylabel('hicp_ROC')
ax2.figure.tight_layout()
ax2.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230103\Stats\HICP_TOTAL_TREND.png', dpi= 600) 
plt.show()

fig, ax3 = plt.subplots(figsize=(10,6))
ax3.set_title("Ireland's Trend in HICP")
sns.lineplot(data=rds[rds['geo']=='IE'], x="date", y="hicp_ROC", color="red", label = 'Ireland')
sns.lineplot(data=rds[rds['geo']!='IE'], x="date", y="hicp_ROC", label = 'Other EU Countries')
ax3.set_xlabel('Month')
ax3.set_ylabel('hicp_ROC')
ax3.legend()
ax3.figure.tight_layout()
ax3.figure.savefig(r'C:\Users\cianw\Documents\dataAnalytics\CA2\Results\20230103\Stats\HICP_IE_TREND.png', dpi= 600) 
plt.show()


rdsIreland=rds[rds['geo']=='IE'] 
rdsOther=rds[rds['geo']!='IE'] 

"""
STATISTICS QUESTION 2
This section will compare populations, such as early Harmonised Risk vs later, early Median incomes, male median incomes vs female, 
"""
"""
Change in Risk
"""
hicp_IE = rdsIreland[['date', 'geo', 'hicp_ROC', 'year_var']]
hicp_Other = rdsOther[['date', 'geo', 'hicp_ROC', 'year_var']]

hicp_IE_Year = hicp_IE.groupby('year_var')['hicp_ROC'].mean().reset_index()
hicp_Other_Year = hicp_Other.groupby(['year_var', 'geo'])['hicp_ROC'].mean().reset_index()

hicp_IE_Year_2009 = hicp_IE_Year[hicp_IE_Year['year_var']==2009]
hicp_Other_Year_2009 = hicp_Other_Year[hicp_Other_Year['year_var']==2009]

hicp_IE_Year_2014 = hicp_IE_Year[hicp_IE_Year['year_var']==2014]
hicp_Other_Year_2014 = hicp_Other_Year[hicp_Other_Year['year_var']==2014]

hicp_IE_Year_2019 = hicp_IE_Year[hicp_IE_Year['year_var']==2019]
hicp_Other_Year_2019 = hicp_Other_Year[hicp_Other_Year['year_var']==2019]

print(shapiro(hicp_Other_Year_2009['hicp_ROC']))
print(kstest(hicp_Other_Year_2009['hicp_ROC'], 'norm'))

#It is clear these are all non normal distributions, lets plot to investigate similarities

sns.histplot(data = hicp_Other_Year_2009 , x = 'hicp_ROC', color="skyblue", label="previous", kde=True)
plt.show()

print()

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























