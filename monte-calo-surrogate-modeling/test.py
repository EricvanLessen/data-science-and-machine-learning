# Splitting data into training and test sets (e.g., 80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#...[train your neural network on X_train and y_train]...

# Predictions on the test set
predictions = model.predict(X_test)

# Calculate Errors on the test set
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared Error on Test Data: {mse}")
print(f"Mean Absolute Error on Test Data: {mae}")
