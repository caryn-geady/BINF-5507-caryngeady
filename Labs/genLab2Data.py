import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Convert to DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
data['target'] = y

# Add some categorical features with inconsistent entries
data['categorical_feature'] = np.random.choice(['A', 'B', 'C', 'A', 'B', 'C', 'NYC', 'SF'], size=1000)

# Introduce missing values
data.loc[data.sample(frac=0.1).index, 'feature_0'] = np.nan
data.loc[data.sample(frac=0.1).index, 'feature_1'] = np.nan

# Introduce some duplicates
data = pd.concat([data, data.sample(frac=0.05)], ignore_index=True)

# Introduce skewed features
data['skewed_feature'] = np.random.exponential(scale=2, size=len(data))

# Save to CSV
data.to_csv('synthetic_data.csv', index=False)