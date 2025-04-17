from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

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


