import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data/Natalia/CSV") 

df = pd.read_csv(data_dir / "aux1_2.csv", sep=";", header=None, names=["time", "signal"])

plt.plot(df["time"], df["signal"])
plt.xlabel("Time")
plt.ylabel("Signal Values")
plt.show()
