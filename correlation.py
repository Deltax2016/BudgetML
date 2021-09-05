import numpy as np
import pandas as pd

df = pd.read_csv('data.csv',sep=",")
df.corr().abs()

# ищем корреляции больше 0.9