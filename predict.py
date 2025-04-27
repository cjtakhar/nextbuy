import pandas as pd
import joblib

# ----------------- Step 1: Load the saved model and encoder -----------------

model = joblib.load("saved_models/xgboost_model.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

print("Model and label encoder loaded successfully.")

# ----------------- Step 2: Create a new fake user feature -----------------

# Simulate a new user with behavior
new_user = {
    "avg_days_between_purchases": 12,  # moderately recent buyer
    "category_Books": 3,
    "category_Clothing": 5,
    "category_Electronics": 2,
    "category_Groceries": 8,
    "category_Home Decor": 1,
    "total_purchases": 19,  # sum of categories
    "recency_bucket": 1     # based on avg_days_between_purchases (0, 1, or 2)
}

# Convert it to a DataFrame
new_user_df = pd.DataFrame([new_user])

print("\nNew user features:")
print(new_user_df)

# ----------------- Step 3: Predict the category -----------------

# Predict
predicted_label = model.predict(new_user_df)[0]

# Decode the numeric label back into the category name
predicted_category = label_encoder.inverse_transform([predicted_label])[0]

print(f"\nPredicted next purchase category: {predicted_category}")
