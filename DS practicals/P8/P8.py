import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("Titanic-Dataset.csv")

features=data.columns
print(features)
for feature in features:

    # pairplot to create a grid of scatterplots, one for each feature specified in vars=[feature]
    # vars=[feature] indicates that only the variable specified in the feature variable will be included in the pairplot.
    sns.pairplot(data, vars=[feature])
    plt.show()

# data['Fare']: Specifies the column 'Fare' from your dataset.
# bins=30: Sets the number of bins in the histogram.
# determines the number of intervals (or bins) into which the data range is divided
# kde=True: Adds a smoothed curve (KDE plot) on top of the histogram.
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()