import numpy as np
import pandas as pd

data=pd.read_csv("Iris.csv")

from sklearn.preprocessing import OrdinalEncoder
d1_reshaped = data['Species'].values.reshape(-1, 1)
encoder = OrdinalEncoder()
encoded_data = encoder.fit_transform(d1_reshaped).astype(int)


df=pd.DataFrame(encoded_data)

data.drop(columns=['Species'], inplace=True)

concatenated_data = pd.concat([data, df], axis=1)

concatenated_data.rename(columns={0: 'new_species'}, inplace=True)

X_ind = concatenated_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y_dep = concatenated_data['new_species']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_ind, Y_dep, test_size=0.2, random_state=42)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score, recall_score,confusion_matrix

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(x_train, y_train)

y_pred = naive_bayes_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)*100
precision =precision_score(y_test, y_pred,average='micro')
recall = recall_score(y_test, y_pred,average='micro') 
cm = confusion_matrix(y_test, y_pred)

print("predicted value:",y_pred)
print("Accuracy:", accuracy,"%")
print("precision:",precision)
print("recall:",recall)
print("confusion matrix:")
print(cm)


