import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("Titanic-Dataset.csv")

# A boxplot visually summarizes numerical data distribution and comparisons across categories
# x: Categorical variable for the x-axis (e.g., 'Sex').
# y: Numerical variable for the y-axis (e.g., 'Age').
# hue: Additional categorical variable for color differentiation (e.g., 'Survived').
# data: DataFrame containing the variables.
sns.boxplot(x='Sex', y='Age', hue='Survived', data=data)
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()


print("Observations:")
print("Age Distribution:")
print("1. The box plot illustrates the distribution of ages for male and female passengers separately.")
print("2. For both genders, the box plots show the central tendency (median) and spread (interquartile range) of ages.")
print("Survival Status:")
print("3. Within each gender group, the box plot distinguishes between passengers who survived and those who did not.")
print("4. It helps visualize any differences in age distribution between survivors and non-survivors within each gender category.")
print("Gender Comparison:")
print("5. The box plot facilitates a comparison of age distributions between males and females.")
print("6. Differences in median age and spread can be observed between the two gender groups.")
print("Outliers:")
print("7. Outliers, represented as individual points beyond the whiskers of the box plot, indicate extreme values in age within each gender and survival category.")