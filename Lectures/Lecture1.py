# Lecture 1 Plots

import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.linspace(0, 10, 10)  # 10 data points
y = np.array([2, 3, 1, 5, 4, 6, 3, 7, 5, 8])  # Example data

# Generate a finer x-axis for smoother plots
x_fine = np.linspace(0, 10, 500)

# Underfitting: Fit a linear model (degree=1)
coeffs_underfit = np.polyfit(x, y, 1)
y_underfit = np.polyval(coeffs_underfit, x_fine)

# Overfitting: Fit a high-degree polynomial (degree=9)
coeffs_overfit = np.polyfit(x, y, 9)
y_overfit = np.polyval(coeffs_overfit, x_fine)

# Optimal Fit: Fit a moderate-degree polynomial (degree=3)
coeffs_optimal = np.polyfit(x, y, 3)
y_optimal = np.polyval(coeffs_optimal, x_fine)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
titles = ['Underfitting', 'Overfitting', 'Optimal Fit']
fits = [y_underfit, y_overfit, y_optimal]

for ax, fit, title in zip(axes, fits, titles):
    ax.scatter(x, y, color='red', edgecolor='black', zorder=5)  # Data points
    ax.plot(x_fine, fit, color='white', zorder=4)  # Fitted line
    ax.set_title(title, color='white', fontsize=14)
    ax.set_facecolor('black')
    ax.grid(False)
    ax.tick_params(colors='white')  # White axis labels
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

plt.tight_layout()
plt.show()
