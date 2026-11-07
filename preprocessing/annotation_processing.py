print("Annotation processing file is running!")

import pandas as pd

file_path = "data/raw/CASE_snippet/data/interpolated/annotations/sub_1.csv"

annotations = pd.read_csv(file_path)

print("\nAnnotations loaded successfully!")
print("Shape:", annotations.shape)
print("Columns:", annotations.columns.tolist())
print("\nFirst 10 rows:")
print(annotations.head(10))


# Check annotation timestamp range
print("\n========== TIMESTAMP CHECK ==========")

print("First annotation timestamp:", annotations["jstime"].min(), "ms")
print("Last annotation timestamp:", annotations["jstime"].max(), "ms")

# Check annotation sampling interval
print(
    "Median annotation interval:",
    annotations["jstime"].diff().dropna().median(),
    "ms"
)

# Check ECG timestamp range
physiology = pd.read_csv(
    "data/raw/CASE_snippet/data/interpolated/physiological/sub_1.csv",
    usecols=["daqtime"]
)

print("\nFirst ECG timestamp:", physiology["daqtime"].min(), "ms")
print("Last ECG timestamp:", physiology["daqtime"].max(), "ms")

# Check video IDs
print("\nVideo IDs:")
print(annotations["video"].value_counts().sort_index())

# Check emotion value ranges
print("\nValence range:",
      annotations["valence"].min(),
      "to", annotations["valence"].max())

print("Arousal range:",
      annotations["arousal"].min(),
      "to", annotations["arousal"].max())


# Check where video segments change

annotations["video_change"] = (
    annotations["video"] != annotations["video"].shift()
)

video_boundaries = annotations.loc[
    annotations["video_change"],
    ["jstime", "video"]
]

print("\n========== VIDEO BOUNDARIES ==========")
print(video_boundaries.to_string(index=False))


# Check video transitions in ECG data

physiology_full = pd.read_csv(
    "data/raw/CASE_snippet/data/interpolated/physiological/sub_1.csv",
    usecols=["daqtime", "video"]
)

physiology_full["video_change"] = (
    physiology_full["video"] != physiology_full["video"].shift()
)

ecg_boundaries = physiology_full.loc[
    physiology_full["video_change"],
    ["daqtime", "video"]
]

print("\n========== ECG VIDEO BOUNDARIES ==========")
print(ecg_boundaries.to_string(index=False))


# Check video transitions in ECG data

physiology_full = pd.read_csv(
    "data/raw/CASE_snippet/data/interpolated/physiological/sub_1.csv",
    usecols=["daqtime", "video"]
)

physiology_full["video_change"] = (
    physiology_full["video"] != physiology_full["video"].shift()
)

ecg_boundaries = physiology_full.loc[
    physiology_full["video_change"],
    ["daqtime", "video"]
]

print("\n========== ECG VIDEO BOUNDARIES ==========")
print(ecg_boundaries.to_string(index=False))



