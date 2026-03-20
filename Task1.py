import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
file_name = "Life Expectancy Data.csv"   
df = pd.read_csv(file_name)


df.columns = df.columns.str.strip()

# Check columns
print("Columns:", df.columns)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Select features
X = df[['Adult Mortality', 'Alcohol', 'GDP', 'Schooling', 'HIV/AIDS']]
y = df['Life expectancy']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
