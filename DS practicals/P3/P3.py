import numpy as np
import pandas as pd

data = pd.read_csv('income.csv')

print(data.head())

# 1
print(np.var(data['Age']))

print(np.mean(data['Age']))

print(np.median(data['Age']))

print(np.min(data['Age']))

print(np.max(data['Age']))

print(np.std(data['Age']))


#2
print(np.var(data['Income']))

print(np.mean(data['Income']))

print(np.median(data['Income']))

print(np.min(data['Income']))

print(np.max(data['Income']))

print(np.std(data['Income']))


categorical_variable = 'Age'  
quantitative_variable  = 'Income'         
summary_stats = data.groupby(categorical_variable)[quantitative_variable].describe()


print(summary_stats.head(5))

print(summary_stats.head(5).sum())

#
# 2nd part

from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['Species'] = iris_data.target_names[iris_data.target]

# Filter the data based on specific species
filtered_data = iris_df[iris_df['Species'].isin(['setosa', 'versicolor', 'virginica'])]

# Calculate summary statistics grouped by species
species_stats = filtered_data.groupby('Species').describe()

print(species_stats)