import numpy as np
import matplotlib.pyplot as plt


def synchronize_lfp_to_emg(t_lfp, lfp_sync_time, emg_sync_time):
    """
    Shift LFP time axis onto EMG time axis.
    """

    t_lfp_sync = t_lfp - lfp_sync_time + emg_sync_time

    return t_lfp_sync


def plot_sync_check(t_emg, emg, t_lfp_sync, lfp, center_time, window=5):
    emg_norm = emg / np.max(np.abs(emg))
    lfp_norm = lfp / np.max(np.abs(lfp))

    plt.figure(figsize=(15, 4))
    plt.plot(t_emg, emg_norm, label="EMG")
    plt.plot(t_lfp_sync, lfp_norm, label="LFP")
    plt.xlim(center_time - window, center_time + window)
    plt.legend()
    plt.title("EMG-LFP synchronization check")
    plt.xlabel("Time [s]")
    plt.show()


def remove_artifacts(t_signal, signal, clean_start, clean_end):
    valid = (t_signal >= clean_start) & (t_signal <= clean_end)

    return t_signal[valid], signal[valid]

def find_emg_sync_after_drop(t_emg, emg, search_start=0, search_end=40):
    """
    Finds EMG synchronization point:
    after the large stimulation drop, where the EMG signal rises again.
    """

    mask = (t_emg >= search_start) & (t_emg <= search_end)
    t = t_emg[mask]
    x = emg[mask]

    # Smooth absolute EMG envelope
    envelope = np.abs(x)
    window = int(0.2 / np.mean(np.diff(t)))  # 200 ms window
    window = max(window, 1)

    kernel = np.ones(window) / window
    smooth_env = np.convolve(envelope, kernel, mode="same")

    # Find biggest negative drop in envelope
    d_env = np.diff(smooth_env)
    drop_idx = np.argmin(d_env)

    # After the drop, find first clear rise
    after = smooth_env[drop_idx:]
    baseline = np.percentile(after, 20)
    high = np.percentile(after, 80)
    threshold = baseline + 0.4 * (high - baseline)

    candidates = np.where(after > threshold)[0]

    if len(candidates) == 0:
        raise ValueError("Could not find EMG sync point automatically.")

    sync_idx_local = drop_idx + candidates[0]
    emg_sync_time = t[sync_idx_local]

    return emg_sync_time


def find_lfp_first_peak(t_lfp, lfp, search_start=None, search_end=None):
    """
    Finds LFP synchronization point:
    first strong peak/deflection in the selected window.
    """

    if search_start is None:
        search_start = t_lfp[0]

    if search_end is None:
        search_end = t_lfp[0] + 30

    mask = (t_lfp >= search_start) & (t_lfp <= search_end)
    t = t_lfp[mask]
    x = lfp[mask]

    abs_x = np.abs(x)

    threshold = np.mean(abs_x) + 5 * np.std(abs_x)

    candidates = np.where(abs_x > threshold)[0]

    if len(candidates) == 0:
        raise ValueError("Could not find LFP peak automatically.")

    lfp_sync_time = t[candidates[0]]

    return lfp_sync_time


def synchronize_lfp_to_emg(t_lfp, lfp_sync_time, emg_sync_time):
    return t_lfp - lfp_sync_time + emg_sync_time


def plot_sync_check(t_emg, emg, t_lfp_sync, lfp, center_time, window=5):
    emg_norm = emg / np.max(np.abs(emg))
    lfp_norm = 1.5 * (lfp / np.max(np.abs(lfp)))

    plt.figure(figsize=(15, 4))
    plt.plot(t_emg, emg_norm, label="EMG")
    plt.plot(t_lfp_sync, lfp_norm, label="LFP")
    plt.xlim(center_time - window, center_time + window)
    plt.legend()
    plt.title("EMG-LFP synchronization check")
    plt.xlabel("Time [s]")
    plt.show()


def remove_artifacts(t_signal, signal, clean_start, clean_end):
    valid = (t_signal >= clean_start) & (t_signal <= clean_end)
    return t_signal[valid], signal[valid]


def select_sync_point(t, signal, title="Select sync point", xlim=None):
    plt.figure(figsize=(15, 4))
    plt.plot(t, signal)

    if xlim is not None:
        plt.xlim(xlim)

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    print(title)
    print("Click one point, then press Enter.")

    point = plt.ginput(1, timeout=0)
    plt.close()

    if len(point) == 0:
        raise ValueError("No point selected.")

    return point[0][0]
