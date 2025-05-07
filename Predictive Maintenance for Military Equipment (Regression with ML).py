import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic dataset for equipment maintenance
data = pd.DataFrame({
    'temperature': np.random.normal(70, 10, 500),  # Sensor: temperature (°C)
    'vibration': np.random.normal(0.5, 0.1, 500),  # Sensor: vibration (mm/s)
    'hours_used': np.random.uniform(100, 1000, 500),  # Operational hours
    'rul': np.random.normal(200, 30, 500)  # Remaining Useful Life (hours)
})
X = data[['temperature', 'vibration', 'hours_used']]
y = data['rul']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced n_estimators for speed
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot predicted vs. actual RUL
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Predicted vs Actual Remaining Useful Life')
plt.tight_layout()
plt.savefig('rul_prediction.png')  # Save for GitHub
plt.show()

# Feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importance:\n", feature_importance)