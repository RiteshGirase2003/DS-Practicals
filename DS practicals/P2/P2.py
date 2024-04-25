import pandas as pd

data = pd.read_csv('tecdiv.csv')

print(data.head())

#1 - Handing missing values 
print(data.isnull().sum())


x = data['First year:   Sem 1'] 
y = data['First year:   Sem 2']

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel('First year:   Sem 1')
plt.ylabel('First year:   Sem 2')
plt.title('Original Non-linear Relationship')
plt.grid(True)
plt.show()

import numpy as np
transformed_x = np.sqrt(x) 
plt.figure(figsize=(8, 6))
plt.scatter(transformed_x, y)
plt.xlabel('First year:   Sem 1')
plt.ylabel('First year:   Sem 2')
plt.title('Transformed Linear Relationship')
plt.grid(True)
plt.show()

# ----------------
import seaborn as sns
from scipy import stats
variable_to_transform = 'Second year:   Sem 1' 
plt.figure(figsize=(10, 4)) 
plt.subplot(1, 2, 1)
sns.histplot(data[variable_to_transform], kde=True)
plt.title('Original Data')


# ------------------
data['log_transformed_variable'] = np.log(data[variable_to_transform] + 1)  
plt.subplot(1, 2, 2)  
sns.histplot(data['log_transformed_variable'], kde=True)
plt.title('Log-Transformed Data')
plt.tight_layout()
plt.show() 


# ----------------------------------------------------------------

# this for first year sem1
from scipy.stats import boxcox

numeric_columns = ['First year:   Sem 1']

transformed_data = pd.DataFrame()
for column in numeric_columns:
    transformed_col, _ = boxcox(data[column] + 1) 
    transformed_data[column] = transformed_col


sns.histplot(transformed_data, kde=True)
plt.show()

transformed_data.to_csv('transformed_tecdiv.csv', index=False)


# this for first year sem2
from scipy.stats import boxcox

numeric_columns = ['First year:   Sem 2']

transformed_data = pd.DataFrame()
for column in numeric_columns:
    transformed_col, _ = boxcox(data[column] + 1) 
    transformed_data[column] = transformed_col


sns.histplot(transformed_data, kde=True)
plt.show()

transformed_data.to_csv('transformed_tecdiv.csv', index=False)


# this for second year sem1
from scipy.stats import boxcox

numeric_columns = ['Second year:   Sem 1']

transformed_data = pd.DataFrame()
for column in numeric_columns:
    transformed_col, _ = boxcox(data[column] + 1) 
    transformed_data[column] = transformed_col


sns.histplot(transformed_data, kde=True)
plt.show()

transformed_data.to_csv('transformed_tecdiv.csv', index=False)

# this for second year sem2
from scipy.stats import boxcox

numeric_columns = ['Second year:   Sem 2']

transformed_data = pd.DataFrame()
for column in numeric_columns:
    transformed_col, _ = boxcox(data[column] + 1) 
    transformed_data[column] = transformed_col


sns.histplot(transformed_data, kde=True)
plt.show()

transformed_data.to_csv('transformed_tecdiv.csv', index=False)