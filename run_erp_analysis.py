import numpy as np
import matplotlib.pyplot as plt


def epoch_signal(t_signal, signal, event_times, tmin=-1.0, tmax=2.0, fs=250):
    n_samples = int((tmax - tmin) * fs)
    epochs = []

    for event_time in event_times:
        mask = (t_signal >= event_time + tmin) & (t_signal < event_time + tmax)
        epoch = signal[mask]

        if len(epoch) == n_samples:
            epochs.append(epoch)

    return np.array(epochs)


blocks = ["Go_Off_1", "Go_Off_2", "Go_Off_3"]

all_epochs = []

for block in blocks:
    data = np.load(f"{block}_preprocessed.npz")

    t_lfp_clean = data["t_lfp_clean"]
    lfp_filtered = data["lfp_filtered"]
    feedback1_times = data["feedback1_times"]
    fs_lfp = float(data["fs_lfp"])

    tmin = -1
    tmax = 2

    epochs = epoch_signal(
        t_lfp_clean,
        lfp_filtered,
        feedback1_times,
        tmin=tmin,
        tmax=tmax,
        fs=fs_lfp
    )

    print(block, epochs.shape)

    all_epochs.append(epochs)

all_epochs = np.vstack(all_epochs)

print("All OFF ERP epochs:", all_epochs.shape)

epoch_time = np.linspace(tmin, tmax, all_epochs.shape[1])

mean_erp = all_epochs.mean(axis=0)
sem_erp = all_epochs.std(axis=0) / np.sqrt(all_epochs.shape[0])

plt.figure(figsize=(10, 5))

plt.plot(epoch_time, mean_erp, label="OFF medication")
plt.fill_between(
    epoch_time,
    mean_erp - sem_erp,
    mean_erp + sem_erp,
    alpha=0.3
)

plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Time from feedback1 [s]")
plt.ylabel("LFP amplitude")
plt.title("OFF medication: Feedback1 ERP")
plt.legend()

plt.savefig("OFF_feedback1_ERP.png", dpi=300)
plt.show()