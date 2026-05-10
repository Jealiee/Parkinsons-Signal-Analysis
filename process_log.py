import pandas as pd
import re


def extract_risk(stim_path):
    if "HR" in stim_path:
        return "high"
    elif "LR" in stim_path:
        return "low"
    return None


def parse_log_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    trials = []
    current = None

    for line in lines:
        line = line.strip()
        trial_match = re.search(r"TRIAL:\s*(\d+)", line)

        if trial_match:
            if current is not None:
                trials.append(current)

            current = {
                "trial": int(trial_match.group(1)),
                "loop": None,
                "symbol1_risk": None,
                "symbol2_risk": None,
                "chosen_symbol": None,
                "feedback": None,
                "alt_feedback": None,
                "symbol1_time": None,
                "symbol2_time": None,
                "choice_time": None,
            }

            continue

        if current is None:
            continue
        final_loop_match = re.search(r"FINAL LOOP:\s*(\d+)", line)
        if final_loop_match:
            current["loop"] = int(final_loop_match.group(1))
            continue
        stim1_match = re.search(r"STIM1:\s*(.+)", line)
        if stim1_match:
            current["symbol1_risk"] = extract_risk(stim1_match.group(1))
            continue

        stim2_match = re.search(r"STIM2:\s*(.+)", line)
        if stim2_match:
            current["symbol2_risk"] = extract_risk(stim2_match.group(1))
            continue

        choice_match = re.search(r"CHOICE POSITION:\s*(first|second)", line)
        if choice_match:
            current["chosen_symbol"] = 1 if choice_match.group(1) == "first" else 2
            continue

        if "NOT WIN:" in line:
            match = re.search(r"NOT WIN:\s*(-?\d+)", line)
            if match:
                current["alt_feedback"] = int(match.group(1))
            continue

        if "WIN:" in line and "NOT WIN" not in line:
            match = re.search(r"WIN:\s*(-?\d+)", line)
            if match:
                current["feedback"] = int(match.group(1))
            continue

    if current is not None:
        trials.append(current)

    return pd.DataFrame(trials)
