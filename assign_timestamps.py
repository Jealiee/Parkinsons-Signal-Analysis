import numpy as np

# Combine the stim ts info with the rest of the df
def assign_stim_timestamps(df, result_df, trials_per_block=50):

    df = df.copy()

    for stim in ["stim1", "stim2"]:
        for loop in range(2):
            df[f"{stim}_ts_loop{loop}"] = np.nan

    stim_blocks = {
        0: {
            "stim1": np.sort(result_df["stim1_f0"].dropna().values),
            "stim2": np.sort(result_df["stim2_f0"].dropna().values),
        },
        1: {
            "stim1": np.sort(result_df["stim1_f1"].dropna().values),
            "stim2": np.sort(result_df["stim2_f1"].dropna().values),
        },
        2: {
            "stim1": np.sort(result_df["stim1_f2"].dropna().values),
            "stim2": np.sort(result_df["stim2_f2"].dropna().values),
        },
    }

    for idx, row in df.iterrows():

        trial = row["trial"]
        block = (trial - 1) // trials_per_block

        trial_start = row["trial_start_ts"]
        trial_end = row["feedback2_ts"]

        if block not in stim_blocks:
            continue

        for stim_name in ["stim1", "stim2"]:

            stim_times = stim_blocks[block][stim_name]

            s = stim_times[
                (stim_times >= trial_start) &
                (stim_times <= trial_end)
            ]

            s = np.sort(s)

            if len(s) > 0:
                df.at[idx, f"{stim_name}_ts_loop0"] = s[0]
            if len(s) > 1:
                df.at[idx, f"{stim_name}_ts_loop1"] = s[1]

    return df