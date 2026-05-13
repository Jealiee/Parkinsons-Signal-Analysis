import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


blocks = ["Go_Off_1", "Go_Off_2", "Go_Off_3"]

summary = []

all_behavior = []

for block in blocks:

    df = pd.read_csv(f"{block}_behavior_with_times.csv")

    mean_rt = df["reaction_time"].mean()

    low_risk_choices = df[
        ((df["chosen_symbol"] == 1) & (df["symbol1_risk"] == "low")) |
        ((df["chosen_symbol"] == 2) & (df["symbol2_risk"] == "low"))
    ]

    low_risk_fraction = len(low_risk_choices) / len(df)

    summary.append({
        "block": block,
        "mean_rt": mean_rt,
        "low_risk_fraction": low_risk_fraction
    })

    all_behavior.append(df)

    print(block)
    print("Mean RT:", round(mean_rt, 3))
    print("Low-risk fraction:", round(low_risk_fraction, 3))
    print()


summary_df = pd.DataFrame(summary)

all_behavior = pd.concat(all_behavior, ignore_index=True)

overall_mean_rt = all_behavior["reaction_time"].mean()

overall_low_risk = len(
    all_behavior[
        ((all_behavior["chosen_symbol"] == 1) & (all_behavior["symbol1_risk"] == "low")) |
        ((all_behavior["chosen_symbol"] == 2) & (all_behavior["symbol2_risk"] == "low"))
    ]
) / len(all_behavior)

print("===== OVERALL OFF =====")
print("Overall mean RT:", round(overall_mean_rt, 3))
print("Overall low-risk fraction:", round(overall_low_risk, 3))


# =========================
# Plot mean RT
# =========================

plt.figure(figsize=(8, 5))

plt.bar(summary_df["block"], summary_df["mean_rt"])

plt.ylabel("Reaction time [s]")
plt.title("Mean reaction time per OFF block")

plt.savefig("OFF_mean_RT.png", dpi=300)
plt.show()


# =========================
# Plot low-risk fraction
# =========================

plt.figure(figsize=(8, 5))

plt.bar(summary_df["block"], summary_df["low_risk_fraction"])

plt.ylabel("Low-risk choice fraction")
plt.title("Low-risk choices per OFF block")

plt.ylim(0, 1)

plt.savefig("OFF_low_risk_choices.png", dpi=300)
plt.show()