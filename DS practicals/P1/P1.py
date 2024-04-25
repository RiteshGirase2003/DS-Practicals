import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# this is to read csv file 
# Load your dataset into a DataFrame (replace 'data.csv' with your actual file name and path)
data = pd.read_csv('vehicles.csv')

#  this is to print first 5 records
print(data.head())
# and if we want to print records as per our need then give count in data.head(10)
# print(data.head(10))

# to check titles of all columns
print(data.columns.tolist())
print(data.columns)



# this is to print all the null values of all features
print(data.isnull().sum())
# to check particular
# print(data['year'].isnull().sum())

# gives you basic statistics like count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile), 75th percentile (Q3), and maximum for numerical columns in a DataFrame.
print(data.describe())





# Get the shape of the DataFrame (number of rows and columns)
shape = data.shape
print("Shape of the DataFrame:", shape)

# Get the number of dimensions of the DataFrame (1 for Series, 2 for DataFrame)
dimensions = data.ndim
print("Number of dimensions of the DataFrame:", dimensions)

# Get concise summary information about the DataFrame, including data types and missing values
info = data.info()
print("Information about the DataFrame:")
print(info)

# Get rows that are duplicates based on all columns in the DataFrame
duplicates = data[data.duplicated()]
print("Duplicate rows in the DataFrame:")
print(duplicates)

# Remove duplicate rows from the DataFrame and update it (inplace=True modifies the DataFrame)
data.drop_duplicates(inplace=True)
print("DataFrame after dropping duplicates:")
print(data)
