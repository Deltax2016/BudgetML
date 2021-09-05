import pandas as pd

df1 = pd.read_csv('data1.csv',sep=",") #ндфл
y1 = df1[:len(df1)-1].iloc[:,-1]
y1.std() # стандартное отклонение

df2 = pd.read_csv('data2.csv',sep=",") # нпо
y2 = df2[:len(df2)-1].iloc[:,-1]
y2.std()

df3 = pd.read_csv('data3.csv',sep=",") # все доходы
y3 = df3[:len(df3)-1].iloc[:,-1]
y3.std()

df4 = pd.read_csv('data4.csv',sep=",") # расходы
y4 = df4[:len(df4)-1].iloc[:,-1]
y4.std()