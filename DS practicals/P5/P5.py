import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

data=pd.read_csv('Social_Network_Ads.csv')




x_ind = data[['Age', 'EstimatedSalary']]  
y_dep = data['Purchased']    
x_train, x_test, y_train, y_test = train_test_split(x_ind, y_dep, test_size=0.2, random_state=42)


logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

# everything is calculated on y_test and y_pred
cm = confusion_matrix(y_test, y_pred)
precession=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
Accuracy=accuracy_score(y_test, y_pred)


print("prediction:",y_pred)
print("confussion matrix:",cm)
print("precession:",precession)
print("recall:",recall)
print("f1:",f1)
print("Accuracy:",Accuracy)



