import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

# iris=load_iris()
# X=iris.data
# y=iris.target
# feature_names=iris.feature_names
# target_names=iris.target_names


# X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#
#
# classifier_knn = KNeighborsClassifier(n_neighbors = 3)
# classifier_knn.fit(X_train, y_train)
# y_pred = classifier_knn.predict(X_test)
# #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# sample = [[5, 5, 3, 2], [2, 4, 3, 5]]
# preds = classifier_knn.predict(sample)
# pred_species = [iris.target_names[p] for p in preds]
# # print("Predictions:", pred_species)
# Input_data = np.array(
#    [[2.1, -1.9, 5.5],
#    [-1.5, 2.4, 3.5],
#    [0.5, -7.9, 5.6],
#    [5.9, 2.3, -5.8]]
# )
# data_binarized=preprocessing.Binarizer(threshold=0.5).transform(Input_data)
# # print("\nBinarized data:\n", data_binarized)
# # print("Mean =", Input_data.mean(axis=0))
# # print("Stddeviation = ", Input_data.std(axis=0))
#
# # data_scaled=preprocessing.scale(Input_data)
# # print(data_scaled.mean(axis=0))
#
# # data_scaler_minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
# # data_scaled_minmax=data_scaler_minmax.fit_transform(Input_data)
# data_normalized_l1=preprocessing.normalize(Input_data, norm="l1")
# data_normalized_l2 = preprocessing.normalize(Input_data, norm='l2')
# # print(data_normalized_l1)
# # print('\n')
# # print(data_normalized_l2)


# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# rng = np.random.RandomState(35)
# x = 10*rng.rand(40)
# y = 2*x-1+rng.randn(40)
#
#
# model=LinearRegression(fit_intercept=True)
# X=x[:, np.newaxis]
# model.fit(X, y)
# xfit=np.linspace(-1, 11)
# Xfit=xfit[:, np.newaxis]
# yfit=model.predict(Xfit)
#
# plt.scatter(x, y)
# plt.plot(Xfit, yfit)
# plt.show()

import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


iris=sns.load_dataset("iris")
X_iris=iris.drop("species", axis=1)
y_iris=iris['species']
model=PCA(n_components=2)
model.fit(X_iris)
#below we are transforming data into 2D
X_2D=model.transform(X_iris)
iris['PCA1']=X_2D[:, 0]
iris['PCA2']=X_2D[:, 1]
sns.scatterplot(x="PCA1", y="PCA2", hue = 'species', data = iris);
plt.show()
# import seaborn as sns; sns.set()
# sns.pairplot(iris, hue='species', height=3);
# plt.show()
