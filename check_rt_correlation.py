import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = r"data\Julia\dataframe\df_full.csv"

log_files = [
    r"data\Julia\Task\Go_Off_1.log",
    r"data\Julia\Task\Go_Off_2.log",
    r"data\Julia\Task\Go_Off_3.log",
]

df = pd.read_csv(csv_path)

csv_rt = df["response_time"].tolist()

all_log_rts = []

rt_pattern = re.compile(r"RT:\s*([0-9.]+)")

for i, log_file in enumerate(log_files, start=1):
    with open(log_file, "r", encoding="utf-8") as f:
        text = f.read()

    matches = rt_pattern.findall(text)

    rts = [float(x) for x in matches]

    print(f"{log_file}: found {len(rts)} RT values")

    if len(rts) != 50:
        raise ValueError(f"{log_file} should contain 50 RTs but found {len(rts)}")

    all_log_rts.extend(rts)


print(f"\nTotal CSV rows: {len(csv_rt)}")
print(f"Total LOG RTs : {len(all_log_rts)}")

if len(csv_rt) != len(all_log_rts):
    raise ValueError("CSV and log RT counts do not match!")

corr = np.corrcoef(csv_rt, all_log_rts)[0, 1]

print("\nPearson Correlation")
print("-------------------")
print(f"r = {corr:.6f}")

comparison = pd.DataFrame(
    {
        "trial": range(1, len(csv_rt) + 1),
        "csv_rt": csv_rt,
        "log_rt": all_log_rts,
        "difference": [c - l for c, l in zip(csv_rt, all_log_rts)],
    }
)

print("\nFirst 10 rows:")
print(comparison.head(10))

plt.figure(figsize=(6, 6))
sns.regplot(x=all_log_rts, y=csv_rt, ci=None)
plt.xlabel("Reaction Time from Log File (s)")
plt.ylabel("Reaction Time from OTB/CSV (s)")

plt.title(f"RT Comparison (r = {corr:.3f})")

plt.tight_layout()
plt.show()
