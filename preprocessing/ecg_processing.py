
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# 1. Load dataset
file_path = "data/raw/CASE_snippet/data/interpolated/physiological/sub_1.csv"
ecg_data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", ecg_data.shape)
print("Column names:", ecg_data.columns.tolist())

# 2. Check sampling frequency
time_difference = ecg_data["daqtime"].diff().dropna()
median_difference = time_difference.median()

sampling_rate = 1000

print("\nMedian time difference:", median_difference, "ms")
print("Estimated sampling frequency:", 1000 / median_difference, "Hz")

# 3. Select first 60 seconds
ecg_sample = ecg_data["ecg"].iloc[:60000].to_numpy()
time = ecg_data["daqtime"].iloc[:60000]

# 4. Clean ECG signal
cleaned_ecg = nk.ecg_clean(
    ecg_sample,
    sampling_rate=sampling_rate,
    method="neurokit"
)

# 5. Plot original and cleaned ECG
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(time, ecg_sample)
plt.title("Original ECG Signal - 60 Seconds")
plt.ylabel("Amplitude (mV)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, cleaned_ecg)
plt.title("Cleaned ECG Signal - 60 Seconds")
plt.xlabel("Time (milliseconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Detect R-peaks
signals, info = nk.ecg_process(
    ecg_sample,
    sampling_rate=sampling_rate
)

rpeaks = info["ECG_R_Peaks"]

print("\nNumber of detected R-peaks:", len(rpeaks))

# 7. Calculate RR intervals
rr_intervals = np.diff(rpeaks) * 1000 / sampling_rate

# 8. Calculate heart rate
heart_rates = 60000 / rr_intervals

print("\nRR Intervals (ms):")
print(rr_intervals)

print("\nHeart Rates (BPM):")
print(heart_rates)

print("\nAverage Heart Rate:",
      round(np.mean(heart_rates), 2), "BPM")

# 9. Plot ECG with R-peaks
plt.figure(figsize=(14, 5))

plt.plot(time, signals["ECG_Clean"], label="Cleaned ECG")
plt.scatter(
    time.iloc[rpeaks],
    signals["ECG_Clean"].iloc[rpeaks],
    color="red",
    label="R-peaks"
)

plt.title("ECG with Detected R-peaks - 60 Seconds")
plt.xlabel("Time (milliseconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 10. Calculate HRV features

mean_rr = np.mean(rr_intervals)
sdnn = np.std(rr_intervals, ddof=1)

successive_differences = np.diff(rr_intervals)
rmssd = np.sqrt(np.mean(successive_differences ** 2))

mean_hr = np.mean(heart_rates)

print("\n========== HRV FEATURES ==========")
print("Mean RR:", round(mean_rr, 2), "ms")
print("SDNN:", round(sdnn, 2), "ms")
print("RMSSD:", round(rmssd, 2), "ms")
print("Mean Heart Rate:", round(mean_hr, 2), "BPM")

