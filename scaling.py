import numpy as np
from sklearn import preprocessing
import pandas as pd
# data={'age':[2,10,30,50,46,70], 'income':[10000,12224,500, 6000, 98,800000], 'ratings':[1,4,3,2,5,2]}
# df=pd.DataFrame(data)
# # print(df.to_string())
# scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
# scaled_data=scaler.fit_transform(data)
# print(scaled_data)


# print(scaled_data)

def scaler_func(dataset):
    scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data

np.random.seed(42)
synthetic_dataset = np.random.rand(5, 4) * np.array([10, 100, 1000, 10000])
scaledd=scaler_func(synthetic_dataset)
print(scaledd)