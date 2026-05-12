import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_signal(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline()

    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(file_path, sep=sep)

    if df.columns[0].startswith("Unnamed") or len(df.columns) == 2 and df.columns[0] != "time":
        df = pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            names=["time", "signal"]
        )
        channels = ["signal"]
    else:
        channels = df.columns[1:]

    plt.figure()

    for ch in channels:
        plt.plot(df["time"], df[ch], label=ch)

    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()

    dt = np.diff(df["time"])
    print("Mean dt:", np.mean(dt))