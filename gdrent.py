import numpy as np
import pandas as pd

# Firstly, we load the dataset
data = pd.read_csv('apartment_price.csv')

# The next step is to extract the features and the target variable
X = data[['SqM', 'Rooms']].values
y = data['Price'].values

# Now, we normalize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Next, we define MSE (the Mean Squared Error) as the cost function
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Now, we define the gradient descent function\n",
def gradient_descent(X, y, lr=0.01, epochs=1000):

    # The initialization of the coefficients
    b = np.zeros(X.shape[1] + 1)

    # We add a column of ones to X for the bias term
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # For a fixed number of epochs, we iterate through the data
    for epoch in range(epochs):
        # The predicted values are calculated
        y_pred = X.dot(b)

        # Calculating the error, using the cost function
        error = mse(y_pred, y)

        # Computing the gradients
        gradients = X.T.dot(y_pred - y) / len(y)

        #Updating the coefficients
        b = b - lr * gradients

        # The error is printed every 100 epochs
        if epoch % 100 == 0:
            print(f'For the epoch {epoch}, we have the following error: Error = {error:.2f}')
    return b

# Next, we train the model using gradient descent
b = gradient_descent(X, y)

# Finally, we print the coefficients
print(f'The coefficients are: b0 = {b[0]:.2f}, b1 = {b[1]:.2f}, b2 = {b[2]:.2f}.')
