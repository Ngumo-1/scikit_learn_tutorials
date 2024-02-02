import numpy as np
from sklearn import preprocessing
import pandas as pd

df=pd.read_csv("dataset.csv")
df.columns=df.columns.str.strip()


binarized_data=preprocessing.Binarizer(threshold=30).transform(df[["Age"]])
print(binarized_data)
#
# def binarizing_func(data):
#
#     column=input("Enter column to be binarized")
#     if column in df.columns:
#         data[column] = preprocessing.Binarizer(threshold=30).transform(df[[column]])
#
#     return data[column]

