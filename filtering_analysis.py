import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch


def bandpass_filter(signal, fs, lowcut=1, highcut=100, order=4):
    b, a = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs
    )

    filtered = filtfilt(b, a, signal)

    return filtered


def plot_psd(raw_signal, filtered_signal, fs):
    f_raw, psd_raw = welch(raw_signal, fs=fs, nperseg=int(fs * 2))
    f_filt, psd_filt = welch(filtered_signal, fs=fs, nperseg=int(fs * 2))

    plt.figure(figsize=(8, 5))
    plt.semilogy(f_raw, psd_raw, label="Raw")
    plt.semilogy(f_filt, psd_filt, label="Filtered")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.title("PSD before and after filtering")
    plt.show()


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


def plot_erp(epochs, tmin=-1.0, tmax=2.0, fs=250, title="ERP"):
    erp = np.mean(epochs, axis=0)
    t_epoch = np.linspace(tmin, tmax, epochs.shape[1])

    plt.figure(figsize=(8, 5))
    plt.plot(t_epoch, erp)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Time from event [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

    return erp, t_epoch
