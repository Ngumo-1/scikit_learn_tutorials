from sklearn import preprocessing
import numpy as np
import pandas as pd

np.random.seed(42)
arr=np.random.rand(3,3)
sq_arr=np.square(arr)
normalised_data=preprocessing.normalize(arr, norm="l2")
print(normalised_data)