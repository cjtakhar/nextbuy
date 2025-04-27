# NextBuy ML Prediction Project

## Overview
NextBuy is a complete end-to-end machine learning project that simulates an e-commerce platform scenario: **predicting a user's next purchase category** based on their past shopping behavior.

The project demonstrates real-world ML practices:
- Synthetic data generation
- Feature engineering from user behavior
- Model training with XGBoost
- Model evaluation (precision, recall, F1)
- Model saving and loading for reuse
- Fast model inference on new data

---

## Problem Statement
Given a user's purchase history across multiple categories, predict the category of their next purchase.  
This simulates business cases like:
- Product recommendation engines
- Personalized shopping experiences
- Customer retention and upselling systems

---

## Project Structure

```
nextbuy/
├── data/
│   └── purchase_history.csv          # Simulated purchase data
├── generate_data.py                  # Script to create and save synthetic user data
├── train_model.py                     # Script for feature engineering, model training, evaluation, saving
├── saved_models/
│   ├── xgboost_model.pkl              # Trained XGBoost model file
│   └── label_encoder.pkl              # LabelEncoder for decoding model outputs
├── predict.py                         # Script to load model and predict new user purchases
├── requirements.txt                   # List of required Python packages
└── README.md                          # Project overview and usage instructions
```

---

## How It Works

### 1. Data Simulation (`generate_data.py`)
- Creates synthetic purchase histories for 100 users.
- Assigns purchases to 5 categories: Electronics, Groceries, Clothing, Books, Home Decor.
- Records "days since last purchase" for added realism.
- Saves generated data to `data/purchase_history.csv`.

### 2. Feature Engineering & Model Training (`train_model.py`)
- One-hot encodes categories.
- Aggregates features at the user level:
  - Average days between purchases
  - Total number of purchases
  - Number of purchases per category
  - Recency bucket (very recent, moderately recent, old)
- Creates target labels based on each user's most recent purchase.
- Splits data into training and test sets (80/20).
- Trains an **XGBoost** classifier.
- Evaluates performance using precision, recall, F1-score, and confusion matrix.
- Saves the trained model and label encoder to `saved_models/`.

### 3. Model Inference (`predict.py`)
- Loads the saved model and label encoder.
- Simulates a new user's feature profile.
- Predicts the most likely next purchase category.
- Decodes the numeric prediction back to the original category name.

---

## Example Output

Running `predict.py`:

```bash
python predict.py
```

Sample output:

```
Model and label encoder loaded successfully.

New user features:
   avg_days_between_purchases  category_Books  ...  total_purchases  recency_bucket
0                          12               3  ...               19               1

Predicted next purchase category: Groceries
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd nextbuy
```

### 2. (Optional) Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

**Step 1: Generate Data**

```bash
python generate_data.py
```

**Step 2: Train and Save the Model**

```bash
python train_model.py
```

**Step 3: Make Predictions**

```bash
python predict.py
```

---

## Key Technologies Used
- Python 3
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

---

## Future Enhancements
- 🚀 Serve the model with an API (FastAPI or Flask)
- 🔍 Hyperparameter tuning for improved model performance
- 📊 Visualizations of performance metrics (confusion matrix heatmaps)
- 🧑‍🧬 Introduce time-decay weighted features
- ☁️ Deploy to a cloud environment (AWS/GCP/Azure)

---

## Author
Built by **Cj Takhar**

---

*Focused on real-world machine learning, product ML workflows, and applied ML engineering.*


