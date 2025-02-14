
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # Corrected import

# Define dataset
data = {
    "Size (sq ft)": [750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400], 
    "Price ($1000s)": [150, 160, 165, 180, 190, 200, 210, 230, 250, 270]
}

df = pd.DataFrame(data)

# Features (X) and Target variable (Y)
X = df[["Size (sq ft)"]]  # Independent variable
y = df["Price ($1000s)"]   # Dependent variable

# Splitting the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Fixed test_size

# Create and train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)  # Fixed function call

# Display results
print("Train set Size:", X_train.shape[0])
print("Test set Size:", X_test.shape[0])
print("Mean Squared Error on Test Set:", mse)
print("Predicted Prices:", y_pred)


# In[ ]:




