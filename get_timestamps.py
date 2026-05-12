import pandas as pd
import numpy as np


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
