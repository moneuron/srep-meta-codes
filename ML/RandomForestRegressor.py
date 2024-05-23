from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load the data
file_path = 'all.csv'
data = pd.read_csv(file_path)
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['cancerType'] = label_encoder.fit_transform(data['cancerType'])
data['coumarin'] = label_encoder.fit_transform(data['coumarin'])

# Select features and target
X = data[['time', 'coumarinDose', 'cancerType', 'coumarin',]]
y = data['viability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error (MSE):', mse)
print('R-squared (R2) Score:', r2)

# Feature Importance for Regression
feature_importances = rf_regressor.feature_importances_
feature_names = X.columns

top_n_features = 8

# Sort feature importances and feature names in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.figure(figsize=(8, 8))
sns.barplot(x=sorted_feature_importances[:top_n_features], y=sorted_feature_names[:top_n_features], palette='viridis')

# Add feature importance values on the plot
for i, v in enumerate(sorted_feature_importances[:top_n_features]):
    plt.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontsize=10)

plt.xlabel('Feature Importance')
plt.title('Top Random Forest Feature Importance (Regression)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error (MSE):', mse)
print('R-squared (R2) Score:', r2)
print('Mean Absolute Error (MAE):', mae)
