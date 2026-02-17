import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -------------------------------
# 1. Load Dataset
# -------------------------------
DATA_PATH = "data/city_day.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found! Check data/city_day.csv path.")

df = pd.read_csv(DATA_PATH)
print("Initial Dataset Shape:", df.shape)

# -------------------------------
# 2. Drop Unnecessary Columns
# -------------------------------
if "City" in df.columns:
    df.drop(columns=["City"], inplace=True)

# -------------------------------
# 3. Handle Date Column (Synchronized)
# -------------------------------
# Since you confirmed the column is 'Datetime'
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.sort_values("Datetime", inplace=True)

# -------------------------------
# 4. Handle Missing Values
# -------------------------------
print("\nMissing values before handling:\n", df.isnull().sum())

# Fill numeric columns with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop rows where AQI (target) or AQI_Bucket (categorical label) are missing
df.dropna(subset=["AQI", "AQI_Bucket"], inplace=True)

print("\nMissing values after handling:\n", df.isnull().sum())

# -------------------------------
# 5. Feature–Target Split (CLEANED)
# -------------------------------
# We remove:
# - AQI: Because it's the target (y)
# - Datetime: Because it's a timestamp object (scalers can't handle objects)
# - AQI_Bucket: Because it's a string/category (scalers can't handle text)
X = df.drop(columns=["AQI", "AQI_Bucket", "Datetime"])
y = df["AQI"]

# -------------------------------
# 6. Train–Test Split (Time-based)
# -------------------------------
split_index = int(len(df) * 0.7)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------------
# 7. Feature Scaling
# -------------------------------
# Now X_train ONLY contains numbers. No strings, no dates.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 8. Save Processed Data
# -------------------------------
OUTPUT_PATH = "data/processed_data.pkl"

joblib.dump(
    {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "scaler": scaler,
        "feature_names": X.columns.tolist()
    },
    OUTPUT_PATH
)

print("\nPreprocessing completed successfully!")
print(f"Processed data saved to: {OUTPUT_PATH}")