# linear_regression_analysis.py

import pandas as pd
import statsmodels.api as sm

# Step 1: Create sample data
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189]
}

# Step 2: Load into DataFrame
df = pd.DataFrame(data)

# Step 3: Define X and y
X = df['YearsExperience']
y = df['Salary']

# Step 4: Add constant to X for the intercept
X = sm.add_constant(X)

# Step 5: Build and fit model
model = sm.OLS(y, X).fit()

# Step 6: Show full summary (Linear Regression Table)
print("📋 Linear Regression Table (Full Summary)")
print(model.summary())

# Step 7: Show regression info
print("\n📌 Regression Info")
print(f"Number of observations: {int(model.nobs)}")
print(f"Degrees of freedom: {int(model.df_model)}")
print(f"AIC: {model.aic}")
print(f"BIC: {model.bic}")

# Step 8: Show regression coefficients
print("\n📊 Regression Coefficients")
print(model.params)

# Step 9: Show p-values
print("\n📉 Regression P-Values")
print(model.pvalues)

# Step 10: Show R-squared
print("\n📈 Regression R-Squared")
print(f"R-squared: {model.rsquared}")
print(f"Adjusted R-squared: {model.rsquared_adj}")
