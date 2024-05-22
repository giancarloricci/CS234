import matplotlib.pyplot as plt
import math

# Parameters
wind_interval = 10000
wind_power = 15  # Example wind power
timesteps = 100000  # Total timesteps to plot

# Generate wind_mag values for the first case
wind_mag_values_case1 = [wind_power * ((i // wind_interval) % 2) for i in range(timesteps)]

# Calculate wind magnitude for each timestep for the second case
wind_mag_values_case2 = []
for wind_idx in range(timesteps):
    normalized_idx = (wind_idx % (4 * wind_interval)) / (4 * wind_interval) * 2 * math.pi
    wind_mag = wind_power * (math.sin(normalized_idx - math.pi / 2) + 1) / 2
    wind_mag_values_case2.append(wind_mag)

# Calculate wind magnitude for each timestep for the third case
wind_mag_values_case3 = []
for wind_idx in range(timesteps):
    wind_mag = (wind_idx % wind_interval) / wind_interval * wind_power
    wind_mag_values_case3.append(wind_mag)

# Calculate wind magnitude for the fourth case (constant wind power)
wind_mag_values_case4 = [wind_power for _ in range(timesteps)]

# Plotting all four cases on the same graph
plt.figure(figsize=(10, 5))

# Case 1
plt.plot(range(timesteps), wind_mag_values_case1, label='Case 1: Step-wise Wind Magnitude')

# Case 2
plt.plot(range(timesteps), wind_mag_values_case2, label='Case 2: Sinusoidal Wind Magnitude')

# Case 3
plt.plot(range(timesteps), wind_mag_values_case3, label='Case 3: Linearly Increasing Wind Magnitude')

# Case 4
plt.plot(range(timesteps), wind_mag_values_case4, label='Case 4: Constant Wind Magnitude')

# Adding labels and title
plt.xlabel('Timestep')
plt.ylabel('Wind Magnitude')
plt.title('Wind Magnitude Over Time for Different Cases')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
