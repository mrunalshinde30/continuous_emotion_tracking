import pandas as pd
import numpy as np

# Load extracted features
file_path = "data/processed/features.csv"
df = pd.read_csv(file_path)

print("\n========== DATASET VALIDATION ==========")

# 1. Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Missing values
print("\n2. Missing Values:")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n3. Duplicate Rows:")
print(df.duplicated().sum())

# 4. Duplicate time windows
print("\n4. Duplicate Windows:")
print(
    df.duplicated(
        subset=["video", "start_time", "end_time"]
    ).sum()
)

# 5. Check infinite values
numeric_columns = df.select_dtypes(include=np.number).columns

print("\n5. Infinite Values:")
print(np.isinf(df[numeric_columns]).sum())

# 6. Feature statistics
print("\n6. Feature Statistics:")
print(
    df[
        ["mean_hr", "mean_rr", "sdnn", "rmssd",
         "valence", "arousal"]
    ].describe()
)

# 7. Screening checks
print("\n7. Screening Checks:")

print(
    "HR below 40 BPM:",
    (df["mean_hr"] < 40).sum()
)

print(
    "HR above 180 BPM:",
    (df["mean_hr"] > 180).sum()
)

print(
    "Non-positive RR:",
    (df["mean_rr"] <= 0).sum()
)

print(
    "Negative SDNN:",
    (df["sdnn"] < 0).sum()
)

print(
    "Negative RMSSD:",
    (df["rmssd"] < 0).sum()
)

# 8. Annotation range checks
print("\n8. Valence and Arousal Range Checks:")

print(
    "Valence outside 0.5–9.5:",
    ((df["valence"] < 0.5) |
     (df["valence"] > 9.5)).sum()
)

print(
    "Arousal outside 0.5–9.5:",
    ((df["arousal"] < 0.5) |
     (df["arousal"] > 9.5)).sum()
)

# 9. Number of windows per video
print("\n9. Windows Per Video:")
print(df["video"].value_counts().sort_index())

# 10. Final result
print("\n========== VALIDATION COMPLETED ==========")

if df.isnull().sum().sum() == 0:
    print("No missing values found.")

if df.duplicated(
    subset=["video", "start_time", "end_time"]
).sum() == 0:
    print("No duplicate time windows found.")

print("\nReview the screening checks before using")
print("the dataset for model training.")