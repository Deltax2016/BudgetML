import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy as pt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing

df = pd.read_csv('data1.csv',sep=",")
#df = pd.read_csv('data2.csv',sep=",") # для НПО
#df = pd.read_csv('data3.csv',sep=",") # для всех налоговых и неналоговых доходов

x = df[:len(df)-1].iloc[:,:-1]
''' для НПО
dat = df.iloc[:,:-1]
dat['dop'] = dat['Скот и птица (в живом весе). тыс.тонн']*dat['Картофель. тыс.тонн']
x = dat[:len(df)-1]
'''
y = df[:len(df)-1].iloc[:,-1]

skm = LinearRegression()
knr = KNeighborsRegressor()
svm = SVR()
rf = RandomForestRegressor()

# запускаем расчет параметров для указанных данных
skm.fit(x, y)
knr.fit(x, y)
svm.fit(x, y)
rf.fit(x, y)

x1 = df[len(df)-1:].iloc[:,:-1]
#x1 = dat[len(df)-1:] для НПО
y1 = df[len(df)-1:].iloc[:,-1]
print('real ', y1) # значение которое должно получиться

#предсказываем
pred_skm = skm.predict(x1)
pred_knr = knr.predict(x1)
pred_svm = svm.predict(x1)
pred_rf = rf.predict(x1)

#выводим результаты и ошибки
print('skm',pred_skm) 
print('knr',pred_knr)
print('svm',pred_svm)
print('rf',pred_rf)
print('skm error',(y1-pred_skm[0])/y1)
print('knr error',(y1-pred_knr[0])/y1)
print('svm error',(y1-pred_svm[0])/y1)
print('rf error',(y1-pred_rf[0])/y1)


