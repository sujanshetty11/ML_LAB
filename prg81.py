from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv("heart.csv")
X=df.drop("sex", axis=1)
y=df['sex']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=42)

clf=GaussianNB()

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
cla=classification_report(y_test,y_pred)

print(f"accuracy :{accuracy * 100 }%" )
print(cla)