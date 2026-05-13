import numpy as np


def detect_trigger_pulses(t, signal, threshold_fraction=0.1, min_interval=0.05):
  
    signal_shifted = signal - np.min(signal)
    threshold = threshold_fraction * np.max(signal_shifted)

    above = signal_shifted > threshold
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    falling_edges = np.where(np.diff(above.astype(int)) == -1)[0] + 1

    event_times = []
    event_amplitudes = []
    edge_indices = []

    last_event_time = -np.inf

    for rise in rising_edges:
        rise_time = t[rise]

        if rise_time - last_event_time < min_interval:
            continue

        fall_candidates = falling_edges[falling_edges > rise]

        if len(fall_candidates) > 0:
            fall = fall_candidates[0]
        else:
            fall = min(rise + 100, len(signal_shifted))

        peak_amp = np.max(signal_shifted[rise:fall])

        event_times.append(rise_time)
        event_amplitudes.append(peak_amp)
        edge_indices.append(rise)

        last_event_time = rise_time

    return (
        np.array(event_times),
        np.array(event_amplitudes),
        np.array(edge_indices),
        signal_shifted
    )


def classify_aux1_events(event_times, event_amplitudes, thresholds):
    thr1, thr2, thr3, thr4 = thresholds

    baseline_times = event_times[event_amplitudes < thr1]

    stim1_times = event_times[
        (event_amplitudes >= thr1) &
        (event_amplitudes < thr2)
    ]

    stim2_times = event_times[
        (event_amplitudes >= thr2) &
        (event_amplitudes < thr3)
    ]

    feedback1_times = event_times[
        (event_amplitudes >= thr3) &
        (event_amplitudes < thr4)
    ]

    feedback2_times = event_times[event_amplitudes >= thr4]

    return {
        "baseline": baseline_times,
        "stim1": stim1_times,
        "stim2": stim2_times,
        "feedback1": feedback1_times,
        "feedback2": feedback2_times,
    }


def validate_feedback_timing(feedback1_times, feedback2_times):
    n = min(len(feedback1_times), len(feedback2_times))

    if n == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "diff": np.array([])
        }

    diff = feedback2_times[:n] - feedback1_times[:n]

    return {
        "mean": np.mean(diff),
        "std": np.std(diff),
        "diff": diff,
    }
