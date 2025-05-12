import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns

data=pd.read_csv("Titanic-Dataset.csv")
print(data.info())
data.drop(["Name","Ticket","Cabin","Embarked"],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Sex']=le.fit_transform(data["Sex"])
print(data.head())


X=data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]
y=data["Survived"]

#DATA PREPROCCESS -- WE FLL EMPTY DATA WITH AVERAGE DATA OF THE COLOMn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
X=imputer.fit_transform(X)
print(data.info())


from sklearn.preprocessing import MinMaxScaler
scler=MinMaxScaler()
scler.fit(X)
scaleddata=scler.transform(X)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaleddata)
dataa=pca.transform(scaleddata)
#shape gives u number of rows and colomns
print("actual data: ", scaleddata.shape)
print("transformed_data : ",dataa.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
 
from sklearn import svm

cls=svm.SVC(kernel="linear")
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)


from sklearn import metrics

accu=metrics.accuracy_score(y_test,y_pred)
print("accuracy = " , accu)

matrix=metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot = True,fmt="d")
mp.title("CONFUSIONN")
mp.xlabel("actual")
mp.ylabel("predicted")
mp.show()