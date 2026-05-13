import pandas as pd
import numpy as np


# Global amplitude thresholding with contiguous segment midpoint detection
def get_response_timestamps(files):
    all_times = []

    for f in files:
        data = pd.read_csv(f)

        t = data["time"].values
        x = data["channel_2"].values

        x = x - np.min(x)

        threshold = 0.5 * np.max(x)

        above = x > threshold

        event_times = []
        in_event = False
        start_idx = None

        for i, val in enumerate(above):
            if val and not in_event:
                start_idx = i
                in_event = True

            elif not val and in_event:
                end_idx = i
                mid_idx = (start_idx + end_idx) // 2
                event_times.append(t[mid_idx])
                in_event = False

        if in_event:
            mid_idx = (start_idx + len(x) - 1) // 2
            event_times.append(t[mid_idx])

        all_times.extend(event_times)
        print(f, len(event_times))

    result_df = pd.DataFrame({"peak_time": all_times})

    return result_df


# Band thresholding with refractory gap + minimum duration filter
def get_event_timestamps(files, min_gap=20, stable_samples=20):

    RANGES = {
        "trial_start": (0.7, 1.3),
        "stim1": (1.35, 1.95),
        "stim2": (2, 2.5),
        "feedback1": (2.75, 3.0),
        "feedback2": (3.25, 3.6),
    }

    pooled_events = {
        "trial_start": [],
        "feedback1": [],
        "feedback2": [],
    }

    stim_events = {
        "stim1": {},
        "stim2": {},
    }

    for f_idx, f in enumerate(files):
        data = pd.read_csv(f)

        t = data.iloc[:, 0].values
        x = data.iloc[:, 1].values

        x = pd.Series(x).rolling(5, center=True, min_periods=1).median().values
        x = np.round(x, 2)

        for event_name, (low, high) in RANGES.items():
            mask = (x >= low) & (x <= high)

            event_times = []
            in_event = False
            start_idx = None
            last_end_idx = -np.inf

            for i, val in enumerate(mask):
                if val and not in_event:
                    if (i - last_end_idx) < min_gap:
                        continue
                    start_idx = i
                    in_event = True

                elif not val and in_event:
                    end_idx = i
                    duration = end_idx - start_idx

                    if duration >= stable_samples:
                        mid_idx = (start_idx + end_idx) // 2
                        event_times.append(t[mid_idx])
                        last_end_idx = end_idx

                    in_event = False

            if in_event:
                end_idx = len(x) - 1
                duration = end_idx - start_idx

                if duration >= stable_samples:
                    mid_idx = (start_idx + end_idx) // 2
                    event_times.append(t[mid_idx])

            if event_name in ["stim1", "stim2"]:
                stim_events[event_name][f"{event_name}_f{f_idx}"] = event_times
            else:
                pooled_events[event_name].extend(event_times)

            print(f"{f} | {event_name}: {len(event_times)}")

    result_dict = {}

    for k, v in pooled_events.items():
        result_dict[k] = pd.Series(v)

    # Separate each file's stim into different columns
    for stim_name, file_dict in stim_events.items():
        for col_name, values in file_dict.items():
            result_dict[col_name] = pd.Series(values)

    result_df = pd.DataFrame(result_dict)

    print("\nTOTAL COUNTS")

    for k in pooled_events:
        print(f"{k}: {len(pooled_events[k])}")

    for stim_name in stim_events:
        total = sum(len(v) for v in stim_events[stim_name].values())
        print(f"{stim_name}: {total}")

    return result_df