#PCA--- Prinvipal component analysis, Preprocessing algorithm
#PCA STEPS
'''-stardaizing the data
-covariance matrix
-finding the solution (aigem)'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
target=data["target"]

data=pd.DataFrame(data["data"], columns=data["feature_names"])
print(data.info())
print(data.head())

from sklearn.preprocessing import MinMaxScaler
scler=MinMaxScaler()
scler.fit(data)
scaleddata=scler.transform(data)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaleddata)
dataa=pca.transform(scaleddata)
#shape gives u number of rows and colomns
print("actual data: ", scaleddata.shape)
print("transformed_data : ",dataa.shape)

mp.figure(figsize=(10,10))
mp.scatter(dataa[:,0],dataa[:,1],c=target)
mp.xlabel("First principal component")
mp.ylabel("second princiapal component")
mp.show()