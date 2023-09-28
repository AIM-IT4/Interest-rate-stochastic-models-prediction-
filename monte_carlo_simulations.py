
import numpy as np
import matplotlib.pyplot as plt

# Simulation setup
initial_rate = 0.03
num_paths = 50
num_steps = 12
dt = 1/12  # monthly steps
time = np.arange(0, 1, dt)

# Vasicek Model Parameters
kappa = 0.5
theta = 0.04
sigma = 0.01

# Monte Carlo simulation for Vasicek Model
rates_vasicek = np.zeros((num_paths, num_steps))
rates_vasicek[:, 0] = initial_rate
for i in range(1, num_steps):
    brownian = np.random.normal(0, np.sqrt(dt), num_paths)
    rates_vasicek[:, i] = rates_vasicek[:, i-1] + kappa * (theta - rates_vasicek[:, i-1]) * dt + sigma * brownian

# ... (similar code for CIR, Hull-White, and BDT models)

# Visualization (subplots for each model)

# ... (subplot code)

plt.show()
