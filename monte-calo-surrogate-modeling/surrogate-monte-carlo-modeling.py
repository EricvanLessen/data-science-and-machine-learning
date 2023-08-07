# pip install pymc3 pandas tensorflow


import pandas as pd
import pymc3 as pm
import numpy as np
import tensorflow as tf

# 1. Load Data
df = pd.read_csv('Binance_1INCHBTC_d.csv')
data = df['Close'].values

# 2. Prepare Sequences
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size-1])
        y.append(data[i+window_size-1])
    return np.array(X), np.array(y)

X, y = create_sequences(data)

# 3. Set Up a Bayesian Model in PyMC3 and Sampling
with pm.Model() as model:
    # Priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('Intercept', 0, sd=20)
    x_coeff = pm.Normal('x_coeff', 0, sd=20, shape=9)
    
    # Expected outcome
    mu = intercept + pm.math.dot(X, x_coeff)
    
    # Likelihood
    price_obs = pm.Normal('price_obs', mu=mu, sd=sigma, observed=y)

    trace = pm.sample(1000, tune=1000, target_accept=0.95)

# 4. Posterior Predictive Checks for surrogate data
ppc = pm.sample_posterior_predictive(trace, model=model, samples=1000)
y_surrogate = np.mean(ppc['price_obs'], axis=0)

# 5. Surrogate Modeling with Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y_surrogate, epochs=50, batch_size=32)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prediction using the surrogate model
predictions = model.predict(X)

# Calculate Errors
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

