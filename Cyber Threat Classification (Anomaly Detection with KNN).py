import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic cybersecurity dataset
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    'packet_size': np.random.normal(100, 20, 1000),
    'duration': np.random.normal(10, 2, 1000),
    'bytes_transferred': np.random.normal(500, 100, 1000),  # Added feature for robustness
    'is_intrusion': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 80% normal, 20% intrusion
})
X = data[['packet_size', 'duration', 'bytes_transferred']]
y = data['is_intrusion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Cyber Threat Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # Save for GitHub
plt.show()

# Save explained variance ratio for PCA (optional insight)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")