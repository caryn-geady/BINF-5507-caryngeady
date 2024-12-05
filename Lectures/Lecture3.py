# Lecture 3 Plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 0.5 * x**2 - 3 * x + 2 + np.random.lognormal(0, 1, size=x.shape)

# Reshape x for sklearn
x = x.reshape(-1, 1)

# Create a high-degree polynomial feature transformer
degree = 15
poly_features = PolynomialFeatures(degree=degree)

# Create models
linear_model = make_pipeline(poly_features, LinearRegression())
ridge_model = make_pipeline(poly_features, Ridge(alpha=1.0))
lasso_model = make_pipeline(poly_features, Lasso(alpha=0.1))
elastic_net_model = make_pipeline(poly_features, ElasticNet(alpha=0.1, l1_ratio=0.5))

# Fit models
linear_model.fit(x, y)
ridge_model.fit(x, y)
lasso_model.fit(x, y)
elastic_net_model.fit(x, y)

# Predict using the models
x_fit = np.linspace(0, 10, 1000).reshape(-1, 1)
y_linear_fit = linear_model.predict(x_fit)
y_ridge_fit = ridge_model.predict(x_fit)
y_lasso_fit = lasso_model.predict(x_fit)
y_elastic_net_fit = elastic_net_model.predict(x_fit)

# Plot the data and the models
plt.figure(figsize=(18, 12))

# Plot original data
# plt.scatter(x, y, color='blue', label='Data')

# Plot overfitting model
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_linear_fit, color='red', linewidth=3,label=f'Overfitting Model (Degree {degree})')
plt.xlabel('Independent Variable', fontsize=18)
plt.ylabel('Dependent Variable', fontsize=18)
plt.title('Overfitting Model', fontsize=24)
plt.legend()

# Plot Ridge model
plt.subplot(2, 2, 2)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_ridge_fit, color='green', linewidth=3,label='Ridge (L2)')
plt.xlabel('Independent Variable', fontsize=18)
plt.ylabel('Dependent Variable', fontsize=18)
plt.title('Ridge Regularization (L2)', fontsize=24)
plt.legend()

# Plot Lasso model
plt.subplot(2, 2, 3)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_lasso_fit, color='purple',linewidth=3, label='Lasso (L1)')
plt.xlabel('Independent Variable', fontsize=18)
plt.ylabel('Dependent Variable', fontsize=18)
plt.title('Lasso Regularization (L1)', fontsize=24)
plt.legend()

# Plot Elastic Net model
plt.subplot(2, 2, 4)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_elastic_net_fit, color='orange', linewidth=3,label='Elastic Net')
plt.xlabel('Independent Variable', fontsize=18)
plt.ylabel('Dependent Variable', fontsize=18)
plt.title('Elastic Net Regularization', fontsize=24)
plt.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()