import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data=pd.read_csv('./housing.csv')



X = data[['RM', 'LSTAT', 'PTRATIO']]
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
#  trains the machine learning model using the provided training data (X_train for features and y_train for target values).
model.fit(X_train, y_train) 
# predict it using the model 
y_pred = model.predict(X_test)

# this create data frames similar to csv
results = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})
print(results)

# model.score(X_test, y_test) is a method in scikit-learn that calculates the accuracy or R-squared score of a trained machine learning model
a=model.score( X_test, y_test)
print("efficency of model :",a*100,"%")

# 
plt.plot(results.head(3) )
plt.axis('tight')
plt.title("Actual Prices vs Predicted Prices")
plt.grid(True)
plt.show()


results.head()