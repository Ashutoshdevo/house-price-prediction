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


