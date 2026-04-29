import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample energy dataset
data = {
    'Day': [1, 2, 3, 4, 5],
    'Energy_Usage': [120, 135, 150, 160, 175]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Day']]
y = df['Energy_Usage']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next day
next_day = [[6]]
prediction = model.predict(next_day)

# Output
print("Smart Building Energy Forecasting")
print("Predicted Energy Usage for Day 6:", round(prediction[0], 2), "kWh")