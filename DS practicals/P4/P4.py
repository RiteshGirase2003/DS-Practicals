import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('housing_data.csv')
data.columns = data.columns.str.strip()
# Separate features and target variable
X = data[['RM', 'LSTAT', 'PTRATIO']]
y = data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Create a DataFrame to compare actual and predicted prices
results = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})

# Calculate the R-squared score
r_squared = model.score(X_test, y_test)
print("Efficiency of the model:", r_squared * 100, "%")

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Actual Prices', y='Predicted Prices', data=results)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.grid(True)
plt.show()

results.head()
