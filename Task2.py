# =====================================
# 1. Import Libraries
# =====================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# 2. Load Data
# =====================================
df = pd.read_csv("Boston.csv")

print("First 5 Rows:\n", df.head())

# =====================================
# 3. Explore Data
# =====================================

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Statistics
print("\nStatistics:\n", df.describe())

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =====================================
# 4. Prepare Data
# =====================================

# Features and Target
X = df[['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']]
y = df['MEDV']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 5. Build Model
# =====================================
model = LinearRegression()
model.fit(X_train, y_train)

# =====================================
# 6. Predictions
# =====================================
y_pred = model.predict(X_test)

# =====================================
# 7. Evaluate Model
# =====================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# =====================================
# 8. Interpretation (Coefficients)
# =====================================
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:\n", coefficients)

# =====================================
# 9. Actual vs Predicted Plot
# =====================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
