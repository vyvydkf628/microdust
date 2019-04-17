# -*- coding: utf-8 -*-
import pandas as pd #Analysis
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis
from scipy.stats import norm #Analysis
from sklearn.preprocessing import StandardScaler #Analysis
from scipy import stats #Analysis
import warnings
warnings.filterwarnings('ignore')
import gc

dust = pd.read_csv('2018.csv', engine='python')
dust.head()
dust.describe()
dust = dust.dropna(axis=0)
dust.describe()

import scipy as sp



dust = dust.drop(['date','pm10'], axis=1)
from sklearn.model_selection import train_test_split
dust_part1, dust_part2 = train_test_split(dust, train_size = 0.8, random_state=3)
print(dust.shape, dust_part1.shape, dust_part2.shape)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

train_columns = [c for c in dust_part1.columns if c not in ['pm25']]
train_columns

from sklearn import linear_model

A = LinearRegression()
A.fit(dust_part1[train_columns], dust_part1['pm25'])
score = A.score(dust_part2[train_columns], dust_part2['pm25'])
print(format(score,'.3f'))

predicted = A.predict(dust_part2[train_columns])
print(predicted.shape)
print(predicted)

dust_part2['price'] = predicted
dust_part2.to_csv('my_submission.csv', index=False)
print('Submission file created!')

from sklearn.ensemble import RandomForestRegressor

B = RandomForestRegressor(n_estimators=28,random_state=0)
B.fit(dust_part1[train_columns], dust_part1['pm25'])
score = B.score(dust_part2[train_columns], dust_part2['pm25'])
print(format(score,'.3f'))

predicted = B.predict(dust_part2[train_columns])
print(predicted.shape)
print(predicted)

dust_part2['price'] = predicted
dust_part2.to_csv('my_submission5.csv', index=False)
print('Submission file created!')

def category_pm10(x):
    if x < 35:
        return 1
    else:
        return 0
dust_part2['pm25_T'] = dust_part2['fact'].apply(category_pm10)
dust_part2['pm25_P'] = dust_part2['predict'].apply(category_pm10)
dust_part2.to_csv('my_submission9.csv', index=False)


