import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("Iris.csv")

feature_types = data.dtypes
print("\nFeatures and their types:")
print(feature_types)

# for feature in data.columns:
#     # sets the size of the plot to 6 inches wide and 4 inches tall.
#     plt.figure(figsize=(6, 4))
#     sns.histplot(data[feature], kde=True)
#     plt.title(f'Histogram of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Count')
#     plt.show()

# it plots boxplot for data
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('Boxplot of Iris Dataset Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=90)  
plt.show()


# Finding outliers 
# An outlier is a data point that is unusually far from the other data points in a dataset.

#  same code as above 

#for SepalLengthCm
Q1 = np.percentile(data['SepalLengthCm'], 25)
Q3 = np.percentile(data['SepalLengthCm'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data['SepalLengthCm'] < lower_bound) | (data['SepalLengthCm'] > upper_bound)
print("for SepalLengthCm")
print("outlier:",outliers)

A=outliers.sum()


#for SepalWidthCm
Q1 = np.percentile(data['SepalWidthCm'], 25)
Q3 = np.percentile(data['SepalWidthCm'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data['SepalWidthCm'] < lower_bound) | (data['SepalWidthCm'] > upper_bound)
print("outlier:",outliers)

B=outliers.sum()


#for PetalLengthCm
Q1 = np.percentile(data['PetalLengthCm'], 25)
Q3 = np.percentile(data['PetalLengthCm'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data['PetalLengthCm'] < lower_bound) | (data['PetalLengthCm'] > upper_bound)
print("outlier:",outliers)



#for PetalWidthCm
Q1 = np.percentile(data['PetalWidthCm'], 25)
Q3 = np.percentile(data['PetalWidthCm'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (data['PetalWidthCm'] < lower_bound) | (data['PetalWidthCm'] > upper_bound)
print("outlier:",outliers)

C=outliers.sum()


if A==0 and B==0 and C==0 :
    print("no outlier in dataset")
else:
    print("outlier is present in dataset") 