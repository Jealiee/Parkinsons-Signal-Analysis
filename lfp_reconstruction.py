import json
import numpy as np
import re


def load_lfp_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data["BrainSenseTimeDomain"]


def parse_numeric_list(x):
    """
    Converts JSON fields to numeric lists.
    Handles:
    - normal Python lists
    - strings like '[64,64,64]'
    - strings like '64,64,64'
    """

    if isinstance(x, list):
        return [float(v) for v in x]

    if isinstance(x, str):
        numbers = re.findall(r"-?\d+\.?\d*", x)
        return [float(v) for v in numbers]

    return [float(x)]


def reconstruct_lfp_time(record):
    fs = float(record["SampleRateInHz"])
    dt = 1 / fs

    lfp = np.array(parse_numeric_list(record["TimeDomainData"]))

    packet_sizes = parse_numeric_list(record["GlobalPacketSizes"])
    ticks_ms = parse_numeric_list(record["TicksInMses"])

    time_parts = []

    for packet_size, tick_ms in zip(packet_sizes, ticks_ms):
        packet_size = int(packet_size)
        tick_ms = float(tick_ms)

        last_time = tick_ms / 1000
        packet_time = last_time - np.arange(packet_size - 1, -1, -1) * dt

        time_parts.append(packet_time)

    t_lfp = np.concatenate(time_parts)

    n = min(len(lfp), len(t_lfp))

    return lfp[:n], t_lfp[:n], fs


def check_lfp_time(t_lfp):
    dt = np.diff(t_lfp)

    return {
        "mean_dt": np.mean(dt),
        "median_dt": np.median(dt),
        "std_dt": np.std(dt),
        "min_dt": np.min(dt),
        "max_dt": np.max(dt),
    }