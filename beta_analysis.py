import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    return filtfilt(b, a, signal)


def epoch_signal(t_signal, signal, event_times, tmin=-1.0, tmax=2.0, fs=250):
    n_samples = int((tmax - tmin) * fs)
    epochs = []

    for event_time in event_times:
        mask = (t_signal >= event_time + tmin) & (t_signal < event_time + tmax)
        epoch = signal[mask]

        if len(epoch) == n_samples:
            epochs.append(epoch)

    return np.array(epochs)


def baseline_zscore_epochs(epochs, epoch_time):
    baseline_mask = epoch_time < 0

    baseline_mean = epochs[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_std = epochs[:, baseline_mask].std(axis=1, keepdims=True)

    return (epochs - baseline_mean) / baseline_std


# =========================
# Load one preprocessed block
# =========================

blocks = ["Go_Off_1", "Go_Off_2", "Go_Off_3"]

all_beta_epochs_z = []

for block in blocks:
    data = np.load(f"{block}_preprocessed.npz")

    t_lfp_clean = data["t_lfp_clean"]
    lfp_filtered = data["lfp_filtered"]
    feedback1_times = data["feedback1_times"]
    fs_lfp = float(data["fs_lfp"])

    beta_signal = bandpass_filter(
        lfp_filtered,
        fs_lfp,
        lowcut=13,
        highcut=30,
        order=4
    )

    analytic_signal = hilbert(beta_signal)
    beta_power = np.abs(analytic_signal) ** 2

    tmin = -1
    tmax = 2

    beta_epochs = epoch_signal(
        t_lfp_clean,
        beta_power,
        feedback1_times,
        tmin=tmin,
        tmax=tmax,
        fs=fs_lfp
    )

    epoch_time = np.linspace(tmin, tmax, beta_epochs.shape[1])

    beta_epochs_z = baseline_zscore_epochs(
        beta_epochs,
        epoch_time
    )

    all_beta_epochs_z.append(beta_epochs_z)

    print(block, beta_epochs_z.shape)

all_beta_epochs_z = np.vstack(all_beta_epochs_z)

print("All OFF beta epochs:", all_beta_epochs_z.shape)

mean_beta = all_beta_epochs_z.mean(axis=0)
sem_beta = all_beta_epochs_z.std(axis=0) / np.sqrt(all_beta_epochs_z.shape[0])

plt.figure(figsize=(10, 5))

plt.plot(epoch_time, mean_beta, label="OFF medication")
plt.fill_between(
    epoch_time,
    mean_beta - sem_beta,
    mean_beta + sem_beta,
    alpha=0.3
)

plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Time from feedback1 [s]")
plt.ylabel("Beta power z-score")
plt.title("OFF medication: Feedback1 beta response")
plt.legend()
plt.savefig("OFF_feedback1_beta_response.png", dpi=300)
plt.show()

