import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("tips.csv")

# --- Visualizations (for insights, not directly part of the model training) ---
# These are the same as your original code, just here for completeness
fig1 = px.scatter(data, x="total_bill", y="tip", size="size", color="day", trendline="ols", title="Tips by Day")
fig2 = px.scatter(data, x="total_bill", y="tip", size="size", color="sex", trendline="ols", title="Tips by Sex")
fig3 = px.scatter(data, x="total_bill", y="tip", size="size", color="time", trendline="ols", title="Tips by Time")
# You'd typically save or display these during development, not necessarily in the training script itself
# if its sole purpose is model training and saving.

# --- Model Training ---
# Encode categorical features
# drop_first=True helps prevent multicollinearity
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features (X) and target (y)
X = data_encoded.drop("tip", axis=1)
y = data_encoded["tip"]

# Save the feature names used during training.
# This is crucial for ensuring consistency when making predictions.
feature_names = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model and the feature names as a dictionary
# This allows you to load both components easily in your Streamlit app
joblib.dump({"model": model, "feature_names": feature_names}, "tip_predictor.pkl")
print("Model trained and saved as 'tip_predictor.pkl' along with feature names.")