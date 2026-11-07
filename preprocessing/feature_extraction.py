
import pandas as pd
import numpy as np
import neurokit2 as nk
import os

# =====================================
# 1. LOAD DATASETS
# =====================================

base_path = "data/raw/CASE_snippet/data/interpolated"

physiology = pd.read_csv(
    f"{base_path}/physiological/sub_1.csv"
)

annotations = pd.read_csv(
    f"{base_path}/annotations/sub_1.csv"
)

print("ECG and annotation datasets loaded!")

# =====================================
# 2. SETTINGS
# =====================================

sampling_rate = 1000
window_size = 60000   # 60 seconds
step_size = 5000      # 5 seconds

features = []

# =====================================
# 3. IDENTIFY VIDEO SEGMENTS
# =====================================

physiology["new_segment"] = (
    physiology["video"] != physiology["video"].shift()
)

segment_starts = physiology.index[
    physiology["new_segment"]
].tolist()

segment_starts.append(len(physiology))

# =====================================
# 4. FEATURE EXTRACTION
# =====================================

for i in range(len(segment_starts) - 1):

    segment = physiology.iloc[
        segment_starts[i]:segment_starts[i + 1]
    ]

    video_id = segment["video"].iloc[0]

    segment_start = int(segment["daqtime"].iloc[0])
    segment_end = int(segment["daqtime"].iloc[-1])

    print(f"Processing video {video_id}...")

    for start in range(
        segment_start,
        segment_end - window_size + 2,
        step_size
    ):

        end = start + window_size

        ecg_window = segment.loc[
            (segment["daqtime"] >= start) &
            (segment["daqtime"] < end),
            "ecg"
        ].to_numpy()

        if len(ecg_window) != window_size:
            continue

        label_window = annotations[
            (annotations["jstime"] >= start) &
            (annotations["jstime"] < end) &
            (annotations["video"] == video_id)
        ]

        if label_window.empty:
            continue

        # Clean ECG
        cleaned_ecg = nk.ecg_clean(
            ecg_window,
            sampling_rate=sampling_rate,
            method="neurokit"
        )

        # Detect R-peaks
        _, info = nk.ecg_peaks(
            cleaned_ecg,
            sampling_rate=sampling_rate
        )

        rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) < 3:
            continue

        # Calculate RR intervals
        rr = np.diff(rpeaks) * 1000 / sampling_rate

        if len(rr) < 2:
            continue

        # Calculate features
        heart_rates = 60000 / rr

        mean_hr = np.mean(heart_rates)
        mean_rr = np.mean(rr)
        sdnn = np.std(rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))

        # Calculate emotion labels
        mean_valence = label_window["valence"].mean()
        mean_arousal = label_window["arousal"].mean()

        features.append({
            "video": video_id,
            "start_time": start,
            "end_time": end,
            "mean_hr": mean_hr,
            "mean_rr": mean_rr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "valence": mean_valence,
            "arousal": mean_arousal
        })

# =====================================
# 5. SAVE FEATURE DATASET
# =====================================

features_df = pd.DataFrame(features)

os.makedirs("data/processed", exist_ok=True)

features_df.to_csv(
    "data/processed/features.csv",
    index=False
)

print("\n========== RESULTS ==========")
print("Total windows:", len(features_df))
print("\nFirst five rows:")
print(features_df.head())

print("\nSaved: data/processed/features.csv")
