from pathlib import Path
import pandas as pd
from process_log import parse_log_file

data_dir = Path("data")

log_files = [
    data_dir / "Task" / "Go_Off_1.log",
    data_dir / "Task" / "Go_Off_2.log",
    data_dir / "Task" / "Go_Off_3.log",
]

dfs = []

for file in log_files:
    df = parse_log_file(file)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
full_df["trial"] = range(1, len(full_df) + 1)

output_folder = data_dir / "parsed_dataframes"
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "Go_Off_full.csv"

full_df.to_csv(output_path, index=False)

print(f"Saved full dataset: {output_path}")
