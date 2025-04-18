from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load California housing dataset
housing = fetch_california_housing()

# Convert it to a pandas DataFrame
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

# Add the target variable (housing prices)
df['Target'] = housing.target

# Show the first few rows of the dataset
print(df.head())

# Check for missing values in the dataset
print(df.isnull().sum())

# Get summary statistics of the dataset
print(df.describe())

# Scatter plot between Median Income (MedInc) and Target (house price)
plt.figure(figsize=(10,6))
sns.scatterplot(x='MedInc', y='Target', data=df)
plt.title('Median Income vs House Price')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

from sklearn.model_selection import train_test_split

# Features (independent variables)
X = df.drop('Target', axis=1)

# Target (dependent variable)
y = df['Target']

# Splitting data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the shape of the splits
print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

from sklearn.linear_model import LinearRegression

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Print the coefficients (weights) for each feature
print(f'Coefficients: {model.coef_}')

# Print the intercept
print(f'Intercept: {model.intercept_}')

# Make predictions using the test data
y_pred = model.predict(X_test)

# Print the first 10 predicted house prices
print(f'Predicted House Prices: {y_pred[:10]}')

from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')


import matplotlib.pyplot as plt

# Predict the values for X_test
y_pred = model.predict(X_test)

# Plotting Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
