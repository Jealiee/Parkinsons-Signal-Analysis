import os
import zipfile
import shutil
import numpy as np
import matplotlib.pyplot as plt

from otb_loader import load_otb
from process_log import parse_log_file

from event_detection import (
    detect_trigger_pulses,
    classify_aux1_events,
    validate_feedback_timing
)
from lfp_reconstruction import (
    load_lfp_json,
    reconstruct_lfp_time,
    check_lfp_time
)
from synchronization import (
    select_sync_point,
    synchronize_lfp_to_emg,
    plot_sync_check,
    remove_artifacts
    
)

from filtering_analysis import (
    bandpass_filter,
    plot_psd,
    epoch_signal,
    plot_erp
)


# =========================
# 1. Extract ZIP
# =========================

zip_path = "Natalia.zip"
extract_dir = "extracted_data"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


# =========================
# 2. Choose block
# =========================

block = "Go_Off_1"

otb_path = fr"extracted_data\Natalia\Natalia\OTB\{block}.otb+"
log_path = fr"extracted_data\Natalia\Natalia\Task\{block}.log"
json_path = fr"extracted_data\Natalia\Natalia\Medtronic\{block}.json"


# =========================
# 3. Load OTB
# =========================

data, t_emg, fs_emg, device_name = load_otb(otb_path)

# DEBUG: plot all OTB channels to identify EMG, AUX1, AUX2
for ch in range(4):
    plt.figure(figsize=(15, 3))
    plt.plot(t_emg, data[ch, :])
    plt.title(f"OTB channel {ch}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
print("Device:", device_name)
print("Sampling frequency EMG:", fs_emg)
print("Data shape:", data.shape)

emg1 = data[0, :]
emg2 = data[1, :]
aux1 = data[2, :]
aux2 = data[3, :]

emg = emg1


# =========================
# 4. Plot raw EMG/AUX
# =========================

plt.figure(figsize=(15, 4))
plt.plot(t_emg, emg)
plt.title("Raw EMG")
plt.xlabel("Time [s]")
plt.show()

plt.figure(figsize=(15, 4))
plt.plot(t_emg, aux1)
plt.title("AUX1 task events")
plt.xlabel("Time [s]")
plt.show()

plt.figure(figsize=(15, 4))
plt.plot(t_emg, aux2)
plt.title("AUX2 responses")
plt.xlabel("Time [s]")
plt.show()


# =========================
# 5. Detect AUX1 events
# =========================

event_times, event_amplitudes, aux1_edges, aux1_shifted = detect_trigger_pulses(
    t_emg,
    aux1,
    threshold_fraction=0.05,
    min_interval=0.05
)

plt.figure(figsize=(15, 4))
plt.plot(t_emg, aux1_shifted)
plt.scatter(event_times, event_amplitudes, color="red")
plt.title("Detected AUX1 rising edges")
plt.show()

plt.figure()
plt.hist(event_amplitudes, bins=50)
plt.title("AUX1 event amplitudes")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.show()


# IMPORTANT:
# Change these thresholds after looking at the histogram.
thresholds = [4500, 7000, 9000, 11000]

events = classify_aux1_events(
    event_times,
    event_amplitudes,
    thresholds
)

baseline_times = events["baseline"]
stim1_times = events["stim1"]
stim2_times = events["stim2"]
feedback1_times = events["feedback1"]
feedback2_times = events["feedback2"]

print("baseline:", len(baseline_times))
print("stim1:", len(stim1_times))
print("stim2:", len(stim2_times))
print("feedback1:", len(feedback1_times))
print("feedback2:", len(feedback2_times))


# =========================
# 6. Detect responses AUX2
# =========================

response_times, response_amplitudes, aux2_edges, aux2_shifted = detect_trigger_pulses(
    t_emg,
    aux2,
    threshold_fraction=0.5,
    min_interval=0.2
)

plt.figure(figsize=(15, 4))
plt.plot(t_emg, aux2_shifted)
plt.scatter(response_times, response_amplitudes, color="red")
plt.title("Detected response rising edges")
plt.show()

print("responses:", len(response_times))


# =========================
# 7. Validate feedback timing
# =========================

validation = validate_feedback_timing(feedback1_times, feedback2_times)

print("Mean feedback2-feedback1:", validation["mean"])
print("STD feedback2-feedback1:", validation["std"])


# =========================
# 8. Parse behavioral log
# =========================

behavior_df = parse_log_file(log_path)

print(behavior_df.head())
print("Number of trials:", len(behavior_df))


# =========================
# 9. Load and reconstruct LFP
# =========================

records = load_lfp_json(json_path)

record = records[0]

lfp, t_lfp, fs_lfp = reconstruct_lfp_time(record)

print("LFP fs:", fs_lfp)
print("LFP length:", len(lfp))

time_check = check_lfp_time(t_lfp)
print(time_check)

# =========================
# 10. Manual synchronization
# =========================

emg_sync_time = select_sync_point(
    t_emg,
    emg,
    title="EMG: click LAST peak before stimulation disappears",
    xlim=(0, 5)
)

lfp_sync_time = select_sync_point(
    t_lfp,
    lfp,
    title="LFP: click FIRST deflection before artifact",
    xlim=(482, 485)
)

print("Selected EMG sync time:", emg_sync_time)
print("Selected LFP sync time:", lfp_sync_time)

t_lfp_sync = synchronize_lfp_to_emg(
    t_lfp,
    lfp_sync_time,
    emg_sync_time
)

plot_sync_check(
    t_emg,
    emg,
    t_lfp_sync,
    lfp,
    center_time=emg_sync_time,
    window=1.5
)
# =========================
# 11. Remove artifacts
# =========================

clean_start = emg_sync_time + 2
clean_end = t_emg[-1] - 2

t_lfp_clean, lfp_clean = remove_artifacts(
    t_lfp_sync,
    lfp,
    clean_start,
    clean_end
)


# =========================
# 12. Filter LFP
# =========================

lfp_filtered = bandpass_filter(
    lfp_clean,
    fs_lfp,
    lowcut=1,
    highcut=100,
    order=4
)


# =========================
# 13. PSD
# =========================

plot_psd(
    lfp_clean,
    lfp_filtered,
    fs_lfp
)


# =========================
# 14. First LFP analysis: feedback1 ERP
# =========================

epochs = epoch_signal(
    t_lfp_clean,
    lfp_filtered,
    feedback1_times,
    tmin=-1,
    tmax=2,
    fs=fs_lfp
)

print("Number of feedback1 epochs:", epochs.shape)

if len(epochs) > 0:
    plot_erp(
        epochs,
        tmin=-1,
        tmax=2,
        fs=fs_lfp,
        title="Feedback1 ERP"
    )