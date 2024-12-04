# Import necessary libraries
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt

# Load the dataset
# The Rossi dataset is a commonly used dataset for survival analysis
# It contains data on the recidivism of released prisoners
data = load_rossi()

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Initialize the Cox Proportional Hazards model
cox_model = CoxPHFitter()

# Fit the model to the data
# The 'week' column is the duration, and 'arrest' is the event of interest
# We include covariates such as 'age', 'race', 'wexp', 'mar', 'paro', and 'prio'
cox_model.fit(data, duration_col='week', event_col='arrest')

# Print the summary of the fitted model
print("\nSummary of the fitted Cox model:")
cox_model.print_summary()

# Make predictions
# We can predict the survival function for each individual
# Here, we predict the survival function for the first 5 individuals
print("\nPredicted survival functions for the first 5 individuals:")
survival_functions = cox_model.predict_survival_function(data.iloc[:5])
print(survival_functions)

# Plot the survival functions
plt.figure(figsize=(10, 6))
for i in range(survival_functions.shape[1]):
    plt.step(survival_functions.index, survival_functions.iloc[:, i], where="post", label=f"Individual {i+1}")
plt.title("Predicted Survival Functions")
plt.xlabel("Time (weeks)")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()
