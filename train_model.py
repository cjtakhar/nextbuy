import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# ----------------- Step 1: Load the Data -----------------

# Load the purchase history data
df = pd.read_csv("data/purchase_history.csv")

print("Sample data:")
print(df.head())

# ----------------- Step 2: Feature Engineering -----------------

# One-hot encode the 'category' for each purchase
df_encoded = pd.get_dummies(df, columns=["category"])

print("\nOne-hot encoded data:")
print(df_encoded.head())

# Group by user and aggregate features
user_features = df_encoded.groupby("user_id").agg({
    "days_since_last_purchase": "mean",  # Average time between purchases
    "category_Books": "sum",
    "category_Clothing": "sum",
    "category_Electronics": "sum",
    "category_Groceries": "sum",
    "category_Home Decor": "sum"
}).reset_index()

# Add total purchases feature
user_features["total_purchases"] = user_features[
    ["category_Books", "category_Clothing", "category_Electronics", "category_Groceries", "category_Home Decor"]
].sum(axis=1)

# Add recency bucket feature
def recency_bucket(x):
    if x <= 10:
        return 0  # very recent buyer
    elif x <= 20:
        return 1  # moderately recent
    else:
        return 2  # older buyer

user_features["recency_bucket"] = user_features["days_since_last_purchase"].apply(recency_bucket)

# Rename for clarity
user_features.rename(columns={"days_since_last_purchase": "avg_days_between_purchases"}, inplace=True)

print("\nUser-level features (with smarter features):")
print(user_features.head())

# ----------------- Step 3: Create Target Labels -----------------

# Goal: Predict the most recent purchase category for each user
latest_purchases = df.sort_values(by="days_since_last_purchase").groupby("user_id").first().reset_index()
targets = latest_purchases[["user_id", "category"]]

# Merge features and targets
dataset = pd.merge(user_features, targets, on="user_id")

print("\nFinal dataset (features + target):")
print(dataset.head())

# Encode target labels (categories) into numeric values
label_encoder = LabelEncoder()
dataset["category_label"] = label_encoder.fit_transform(dataset["category"])

# ----------------- Step 4: Train/Test Split -----------------

X = dataset.drop(columns=["user_id", "category", "category_label"])
y = dataset["category_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

# ----------------- Step 5: Train the Model (XGBoost) -----------------

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"  # multiclass log-loss
)

model.fit(X_train, y_train)

# ----------------- Step 6: Predict and Evaluate -----------------

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------- Step 7: Save the Model and Label Encoder -----------------

# Create folder to save model if it doesn't exist
os.makedirs("saved_models", exist_ok=True)

# Save the trained model
model_path = "saved_models/xgboost_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Save the label encoder
encoder_path = "saved_models/label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)
print(f"Label encoder saved to {encoder_path}")
