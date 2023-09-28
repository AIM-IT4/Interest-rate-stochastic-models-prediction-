
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
# Cox-Ingersoll-Ross (CIR) Model Parameters (using similar parameters with adjustments)
kappa_cir = 0.5
theta_cir = 0.04
sigma_cir = 0.02
#BDT Parameters 
kappa_bdt = 0.5
theta_bdt = 0.04
sigma_bdt = 0.01
# Hull-White Model Parameters (using similar parameters to Vasicek for simplicity)
kappa_hw = 0.5
theta_hw = 0.04
sigma_hw = 0.01


# Vasicek Model Simulation
rates_vasicek_1yr = np.zeros((num_paths, num_steps))
rates_vasicek_1yr[:, 0] = initial_rate
for i in range(1, num_steps):
    brownian = np.random.normal(0, np.sqrt(dt), num_paths)
    rates_vasicek_1yr[:, i] = rates_vasicek_1yr[:, i-1] + kappa * (theta - rates_vasicek_1yr[:, i-1]) * dt + sigma * brownian

# CIR Model Simulation
rates_cir_1yr = np.zeros((num_paths, num_steps))
rates_cir_1yr[:, 0] = initial_rate
for i in range(1, num_steps):
    brownian = np.random.normal(0, np.sqrt(dt), num_paths)
    rates_cir_1yr[:, i] = rates_cir_1yr[:, i-1] + kappa_cir * (theta_cir - rates_cir_1yr[:, i-1]) * dt + sigma_cir * np.sqrt(rates_cir_1yr[:, i-1]) * brownian
rates_cir_1yr = np.maximum(rates_cir_1yr, 0)

# Hull-White Model Simulation
rates_hw_1yr = np.zeros((num_paths, num_steps))
rates_hw_1yr[:, 0] = initial_rate
for i in range(1, num_steps):
    brownian = np.random.normal(0, np.sqrt(dt), num_paths)
    rates_hw_1yr[:, i] = rates_hw_1yr[:, i-1] + kappa_hw * (theta_hw - rates_hw_1yr[:, i-1]) * dt + sigma_hw * brownian

# Approximated BDT Model Simulation
rates_bdt_1yr = np.zeros((num_paths, num_steps))
rates_bdt_1yr[:, 0] = initial_rate
for i in range(1, num_steps):
    brownian = np.random.normal(0, np.sqrt(dt), num_paths)
    rates_bdt_1yr[:, i] = rates_bdt_1yr[:, i-1] + kappa_bdt * (theta_bdt - rates_bdt_1yr[:, i-1]) * dt + sigma_bdt * brownian

# Create subplots for all the models over 1 year
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot Vasicek Model simulations
for i in range(num_paths):
    axs[0, 0].plot(time, rates_vasicek_1yr[i, :], lw=0.8)
axs[0, 0].set_title('Vasicek Model')
axs[0, 0].set_ylabel('Interest Rate')
axs[0, 0].grid(True)

# Plot CIR Model simulations
for i in range(num_paths):
    axs[0, 1].plot(time, rates_cir_1yr[i, :], lw=0.8)
axs[0, 1].set_title('Cox-Ingersoll-Ross (CIR) Model')
axs[0, 1].grid(True)

# Plot Hull-White Model simulations
for i in range(num_paths):
    axs[1, 0].plot(time, rates_hw_1yr[i, :], lw=0.8)
axs[1, 0].set_title('Hull-White Model')
axs[1, 0].set_xlabel('Years')
axs[1, 0].set_ylabel('Interest Rate')
axs[1, 0].grid(True)

# Plot BDT Model simulations
for i in range(num_paths):
    axs[1, 1].plot(time, rates_bdt_1yr[i, :], lw=0.8)
axs[1, 1].set_title('Black-Derman-Toy (BDT) Model (Approximated)')
axs[1, 1].set_xlabel('Years')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

#Approx values 

# Extract predicted rates at the end of the next month and compute mean and standard deviation
mean_vasicek = np.mean(rates_vasicek_1yr[:, 1])
std_vasicek = np.std(rates_vasicek_1yr[:, 1])

mean_cir = np.mean(rates_cir_1yr[:, 1])
std_cir = np.std(rates_cir_1yr[:, 1])

mean_hw = np.mean(rates_hw_1yr[:, 1])
std_hw = np.std(rates_hw_1yr[:, 1])

mean_bdt = np.mean(rates_bdt_1yr[:, 1])
std_bdt = np.std(rates_bdt_1yr[:, 1])

mean_vasicek, std_vasicek, mean_cir, std_cir, mean_hw, std_hw, mean_bdt, std_bdt
