import pandas as pd
import numpy as np
import os

# Set a seed for reproducibility
np.random.seed(42)

# Simulate some users
user_ids = [f"user_{i}" for i in range(1, 101)]  # 100 users

# Define some product categories
categories = ["Electronics", "Groceries", "Clothing", "Books", "Home Decor"]

# Create fake purchase history
data = []
for user in user_ids:
    num_purchases = np.random.randint(5, 20)  # Each user made between 5 and 20 purchases
    for _ in range(num_purchases):
        purchase = {
            "user_id": user,
            "category": np.random.choice(categories),
            "days_since_last_purchase": np.random.randint(1, 30)
        }
        data.append(purchase)

# Create a DataFrame
df = pd.DataFrame(data)

# Create a 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the DataFrame as a CSV file
csv_path = os.path.join("data", "purchase_history.csv")
df.to_csv(csv_path, index=False)

print(f"Dataset saved to {csv_path}")
print(df.head())
