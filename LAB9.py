import pandas as pd
import numpy as np
import streamlit as st
from math import ceil
from scipy import linalg
import matplotlib.pyplot as plt

# Function to perform LOWESS smoothing
def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)], [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

# Streamlit UI elements
st.title('LOWESS Smoothing with Streamlit')
st.write('This application performs LOWESS smoothing on a noisy sine wave.')

# Parameters for LOWESS
f = st.slider('Smoothing factor (f)', 0.1, 1.0, 0.25)
iterations = st.slider('Number of iterations', 1, 10, 3)

# Generate data
n = 100
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)

# Perform LOWESS smoothing
yest = lowess(x, y, f, iterations)

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y, 'r.', label='Noisy Data')
ax.plot(x, yest, 'b-', label='LOWESS Smoothed')
ax.legend()
st.pyplot(fig)
