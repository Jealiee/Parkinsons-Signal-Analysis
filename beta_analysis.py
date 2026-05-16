import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import glob

# =========================
# Settings
# =========================

patient = "Natalia"   # change to "Natalia" if needed
condition = "Off"
blocks = ["Go_Off_1", "Go_Off_2", "Go_Off_3"]

input_folder = "."  # folder where *_preprocessed.npz files are
output_folder = f"beta_results_{patient}_{condition}"

os.makedirs(output_folder, exist_ok=True)


# =========================
# Helper functions
# =========================

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    b, a = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs
    )
    return filtfilt(b, a, signal)


def epoch_signal(t_signal, signal, event_times, tmin=-1.0, tmax=2.0, fs=250):
    n_samples = int((tmax - tmin) * fs)
    epochs = []

    for event_time in event_times:
        start = event_time + tmin
        end = event_time + tmax

        mask = (t_signal >= start) & (t_signal < end)
        epoch = signal[mask]

        if len(epoch) == n_samples:
            epochs.append(epoch)

    return np.array(epochs)


def baseline_zscore_epochs(epochs, epoch_time):
    baseline_mask = epoch_time < 0

    baseline_mean = epochs[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_std = epochs[:, baseline_mask].std(axis=1, keepdims=True)

    # avoid division by zero
    baseline_std[baseline_std == 0] = np.nan

    return (epochs - baseline_mean) / baseline_std


# =========================
# Beta analysis
# =========================

tmin = -1
tmax = 2

all_beta_epochs_z = []
summary_rows = []

for block in blocks:

    possible_patterns = [
    os.path.join(input_folder, f"{patient}_{block}_preprocessed.npz"),
    os.path.join(input_folder, f"{block}_preprocessed.npz"),
    os.path.join(input_folder, "**", f"{patient}_{block}_preprocessed.npz"),
    os.path.join(input_folder, "**", f"{block}_preprocessed.npz"),
    ]

    matches = []

    for pattern in possible_patterns:
        matches.extend(glob.glob(pattern, recursive=True))

    matches = list(set(matches))

    if len(matches) == 0:
        print(f"Missing file for {block}. Tried patterns:")
        for pattern in possible_patterns:
            print("  ", pattern)
        continue

    file_path = matches[0]
    print("Using file:", file_path)

    if file_path is None:
        print(f"Missing file for {block}. Tried:")

        for name in possible_names:
            print("  ", os.path.join(input_folder, name))

        continue

    print("Using file:", file_path)


    data = np.load(file_path)

    t_lfp_clean = data["t_lfp_clean"]
    lfp_filtered = data["lfp_filtered"]
    feedback1_times = np.array(data["feedback1_times"])
    fs_lfp = float(data["fs_lfp"])

    print("LFP clean range:", t_lfp_clean[0], "to", t_lfp_clean[-1])
    print("Feedback1 range:", feedback1_times[0], "to", feedback1_times[-1])
    print("Number feedback1 before filtering:", len(feedback1_times))

    # Keep only events that fit inside clean LFP window
    valid_feedback1 = feedback1_times[
        (feedback1_times > t_lfp_clean[0] - tmin) &
        (feedback1_times < t_lfp_clean[-1] - tmax)
    ]

    print("Valid feedback1 events:", len(valid_feedback1))

    if len(valid_feedback1) == 0:
        print(f"No valid feedback1 events for {block}. Skipping.")
        continue

    beta_signal = bandpass_filter(
        lfp_filtered,
        fs_lfp,
        lowcut=13,
        highcut=30,
        order=4
    )

    analytic_signal = hilbert(beta_signal)
    beta_power = np.abs(analytic_signal) ** 2

    beta_epochs = epoch_signal(
        t_lfp_clean,
        beta_power,
        valid_feedback1,
        tmin=tmin,
        tmax=tmax,
        fs=fs_lfp
    )

    print("Beta epochs:", beta_epochs.shape)

    if beta_epochs.shape[0] == 0:
        print(f"No complete beta epochs for {block}. Skipping.")
        continue

    epoch_time = np.arange(beta_epochs.shape[1]) / fs_lfp + tmin

    beta_epochs_z = baseline_zscore_epochs(
        beta_epochs,
        epoch_time
    )

    # remove epochs with NaN
    beta_epochs_z = beta_epochs_z[~np.isnan(beta_epochs_z).any(axis=1)]

    print("Beta epochs after NaN removal:", beta_epochs_z.shape)

    if beta_epochs_z.shape[0] == 0:
        print(f"All beta epochs invalid for {block}. Skipping.")
        continue

    all_beta_epochs_z.append(beta_epochs_z)

    mean_beta_block = beta_epochs_z.mean(axis=0)
    sem_beta_block = beta_epochs_z.std(axis=0) / np.sqrt(beta_epochs_z.shape[0])

    # Save block data
    np.savez(
        os.path.join(output_folder, f"{patient}_{block}_beta_epochs.npz"),
        beta_epochs_z=beta_epochs_z,
        mean_beta=mean_beta_block,
        sem_beta=sem_beta_block,
        epoch_time=epoch_time,
        valid_feedback1=valid_feedback1,
        fs_lfp=fs_lfp
    )

    # Plot block beta
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_time, mean_beta_block, label=f"{patient} {block}")
    plt.fill_between(
        epoch_time,
        mean_beta_block - sem_beta_block,
        mean_beta_block + sem_beta_block,
        alpha=0.3
    )
    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Time from Feedback1 [s]")
    plt.ylabel("Beta power z-score")
    plt.title(f"{patient} {block}: Feedback1 beta response")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_folder, f"{patient}_{block}_feedback1_beta_response.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    summary_rows.append([
        patient,
        block,
        beta_epochs_z.shape[0],
        np.nanmean(mean_beta_block),
        np.nanmax(mean_beta_block),
        epoch_time[np.nanargmax(mean_beta_block)]
    ])


# =========================
# Combined OFF beta response
# =========================

if len(all_beta_epochs_z) > 0:

    all_beta_epochs_z = np.vstack(all_beta_epochs_z)

    mean_beta = all_beta_epochs_z.mean(axis=0)
    sem_beta = all_beta_epochs_z.std(axis=0) / np.sqrt(all_beta_epochs_z.shape[0])

    print("\nAll beta epochs:", all_beta_epochs_z.shape)

    np.savez(
        os.path.join(output_folder, f"{patient}_{condition}_all_feedback1_beta_response.npz"),
        all_beta_epochs_z=all_beta_epochs_z,
        mean_beta=mean_beta,
        sem_beta=sem_beta,
        epoch_time=epoch_time
    )

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_time, mean_beta, label=f"{patient} {condition} medication")
    plt.fill_between(
        epoch_time,
        mean_beta - sem_beta,
        mean_beta + sem_beta,
        alpha=0.3
    )

    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Time from Feedback1 [s]")
    plt.ylabel("Beta power z-score")
    plt.title(f"{patient} {condition}: Feedback1 beta response")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(output_folder, f"{patient}_{condition}_feedback1_beta_response.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

else:
    print("\nNo beta epochs found for any block.")


# =========================
# Save summary CSV
# =========================

summary_path = os.path.join(output_folder, f"{patient}_{condition}_beta_summary.csv")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("patient,block,n_epochs,mean_beta_z,max_beta_z,time_of_max_beta_s\n")

    for row in summary_rows:
        f.write(",".join(map(str, row)) + "\n")

print(f"\nSaved beta summary: {summary_path}")
print(f"Saved beta results in folder: {output_folder}")
