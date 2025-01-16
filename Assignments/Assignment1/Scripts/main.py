import pandas as pd
from DataPreprocessor import DataPreprocessor

# Load the dataset
data = pd.read_csv('Data/messy_data.csv')

# Initialize the preprocessor
preprocessor = DataPreprocessor(data)

# Apply preprocessing steps
preprocessor.handle_duplicates()
preprocessor.impute_missing_values(strategy='mean')
preprocessor.remove_redundant_features()
preprocessor.normalize_data()
preprocessor.encode_categorical_features()
preprocessor.detect_and_handle_outliers()

# Save the cleaned data
cleaned_data = preprocessor.data
cleaned_data.to_csv('Data/cleaned_data.csv', index=False)
