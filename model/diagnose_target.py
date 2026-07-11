# model/diagnose_target.py
import glob
import os
import csv
import pandas as pd
import numpy as np

DATA_DIR = "model/data"
OUT_CSV = "model/diagnostics_summary.csv"
TARGET = "pmjdy_per_1000_pop"
TOP_N = 20


def summarize_file(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        return None
    s = df[TARGET].describe()
    zeros = (df[TARGET] == 0).sum()
    pct_zeros = zeros / len(df) * 100
    top = df[[TARGET]].sort_values(by=TARGET, ascending=False).head(TOP_N)
    return {
        "file": os.path.basename(path),
        "count": int(s["count"]),
        "mean": float(s["mean"]),
        "std": float(s["std"]),
        "min": float(s["min"]),
        "25%": float(s["25%"]),
        "50%": float(s["50%"]),
        "75%": float(s["75%"]),
        "max": float(s["max"]),
        "zeros": int(zeros),
        "pct_zeros": float(pct_zeros),
        "top_values": ";".join(map(str, top[TARGET].values.tolist()))
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_dataset.csv")))
    rows = []
    print(f"Found {len(files)} files in {DATA_DIR}")
    for p in files:
        info = summarize_file(p)
        if info:
            rows.append(info)
            print(f"{info['file']}: count={info['count']}, mean={info['mean']:.2f}, std={info['std']:.2f}, min={info['min']}, median={info['50%']}, max={info['max']}, zeros={info['zeros']} ({info['pct_zeros']:.2f}%)")
        else:
            print(f"{os.path.basename(p)}: target {TARGET} not found, skipping.")

    # Save summary CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote diagnostics summary to {OUT_CSV}")

    # Print a short global summary (concatenate)
    all_vals = []
    for p in files:
        df = pd.read_csv(p)
        if TARGET in df.columns:
            all_vals.extend(df[TARGET].fillna(0).values.tolist())
    if all_vals:
        arr = np.array(all_vals, dtype=float)
        print("\nGlobal combined target stats:")
        print(f"rows: {len(arr)}, mean: {arr.mean():.2f}, std: {arr.std():.2f}, min: {arr.min()}, median: {np.median(arr)}, max: {arr.max()}")
        print("Global percentiles: ", np.percentile(
            arr, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]))


if __name__ == '__main__':
    main()
