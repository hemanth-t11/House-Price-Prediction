import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("train.csv")

# Select features you want to use in the Streamlit app
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
X = df[selected_features]
y = df["SalePrice"]

# Handle missing values
X.fillna(X.median(numeric_only=True), inplace=True)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and feature list
joblib.dump(model, "house_price_model.pkl")
joblib.dump(selected_features, "features.pkl")

print("Model trained and saved!")
