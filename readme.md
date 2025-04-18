# House Price Prediction

This is a simple Machine Learning project that predicts house prices using the California Housing Dataset.

## Features Used
- Median Income (MedInc)
- House Age
- Average Rooms (AveRooms)
- Average Bedrooms (AveBedrms)
- Population
- Average Occupancy (AveOccup)
- Latitude
- Longitude

## Workflow
1. Loaded and explored the dataset
2. Preprocessed the data
3. Split into training and testing sets
4. Trained a Linear Regression model
5. Evaluated model performance using:
   - Mean Squared Error (MSE)
   - R² Score
6. Visualized Actual vs Predicted Prices
7. Saved the trained model using Pickle

## Results
- **MSE:** 0.5558
- **R² Score:** 0.5757

## Tools Used
- Python
- scikit-learn
- Pandas
- Numpy
- Matplotlib
- Pickle

📦 Model Saving Using Pickle
After training the model, we saved it using the pickle library.
This allows the trained model to be reused later without retraining, making predictions faster and easier.

Steps followed:

Saving the model:
We serialized the trained Linear Regression model and saved it into a file named model.pkl.

Benefits:

No need to retrain every time

Easy deployment into web apps

Faster and more efficient predictions

## Author
- [Ashutosh](https://github.com/Ashutoshdevo)
