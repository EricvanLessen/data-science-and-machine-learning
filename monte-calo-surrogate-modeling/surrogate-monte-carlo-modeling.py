import tensorflow as tf

import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro

import pyro.distributions as dist

from pyro.infer import MCMC, NUTS

# 1. Load Data
df = pd.read_csv('Binance_1INCHBTC_d.csv', skiprows=1)
df.head()

# 1. Load the data 
close_prices = df['Close'].values
close_prices[0:10]

# 2. Prepare sequences
window_size = 10
sequences = [close_prices[i: i + window_size] for i in range(len(close_prices) - window_size + 1)]

# 3. Markov Chain using Pyro

def markov_model(data):
    mu = pyro.sample('mu', dist.Normal(0, 10))
    sigma = pyro.sample('sigma', dist.HalfNormal(10))
    for i, value in enumerate(data):
        pyro.sample(f"data_{i}", dist.Normal(mu, sigma), obs=value)

def get_mcmc_samples(data, num_samples=1000, warmup_steps=200):
    nuts_kernel = NUTS(markov_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(torch.tensor(data))
    return mcmc.get_samples()

def predict_next_value(data):
    samples = get_mcmc_samples(data[:-1])
    predicted_mu = samples['mu'].mean().item()
    return predicted_mu

# 4. Create training data using MCMC predictions
X = [seq[:9] for seq in sequences]
y = [predict_next_value(seq) for seq in sequences]

# 5. Surrogate modeling using Neural Network

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(np.array(X), np.array(y), epochs=50, batch_size=32)

# 6. Evaluate the model

# Splitting data into train and test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate MSE and MAE
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

mse_value = mse(y_test, y_pred).numpy()
mae_value = mae(y_test, y_pred).numpy()

print(f"MSE: {mse_value}, MAE: {mae_value}")