import pandas as pd
import matplotlib.pyplot as plt


file_path = "data\Julia\CSV\events_1.csv"

df = pd.read_csv(file_path)

time = df.iloc[:, 0].values
signal = df.iloc[:, 1].values

plt.figure(figsize=(18, 6))

plt.plot(time, signal, linewidth=1)

plt.title("Channel 1 Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()
