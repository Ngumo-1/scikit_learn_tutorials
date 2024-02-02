import numpy as np
from sklearn import preprocessing
import pandas as pd


def func_to_remove_mean(data):
    data_removed_mean = preprocessing.scale(data)
    return data_removed_mean

np.random.seed(42)
exam_scores = np.random.randint(60, 100, size=(5, 4)) # 5 students, 4 exams
data=pd.DataFrame(exam_scores)
data_to_transform=data.loc[0:4]
results=func_to_remove_mean(data_to_transform)
print(results)