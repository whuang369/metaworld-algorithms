#!/usr/bin/env python
"""
Download full metric curves for all runs in a W&B group, and save each run
individually as a .npz file under results/<group>/.

Example:

python download_group_runs_to_npz.py \
  --entity whuang369-university-of-wisconsin-madison \
  --project metaworld_reproduce \
  --group baseline/ppo_mt10 \
  --metric eval/success_rate
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True,
                        help="W&B entity (user or team)")
    parser.add_argument("--project", type=str, required=True,
                        help="W&B project name")
    parser.add_argument("--group", type=str, required=True,
                        help="W&B group name (e.g. baseline/ppo_mt10)")
    parser.add_argument("--metric", type=str, required=True,
                        help="Metric key in W&B history (e.g. eval/success_rate)")
    parser.add_argument("--output-root", type=str, default="results",
                        help="Root directory for saving results (default: results)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional: maximum number of rows to load from history")
    return parser.parse_args()


def sanitize_key(s: str) -> str:
    """
    Make a metric name safe to use in a numpy .npz key.
    For example: 'eval/success_rate' -> 'eval_success_rate'
    """
    return re.sub(r"[^0-9a-zA-Z_]+", "_", s)


def load_run_history(run, metric, max_rows=None):
    """
    Return (step, values) for a scalar metric.

    Tries to automatically find a suitable step column among:
    'step', 'global_step', '_step', 'Step'.
    """
    # 1) Get plenty of history rows
    samples = max_rows if max_rows is not None else 100000
    hist = run.history(samples=samples)

    # Just in case wandb returns a list
    if isinstance(hist, list):
        hist = pd.DataFrame(hist)

    # print("History columns:", list(hist.columns))

    if metric not in hist.columns:
        print(f"Metric '{metric}' not found in history.")
        return None

    # 2) Find a step-like column
    step_key = None
    for cand in ["Step"]:
        if cand in hist.columns:
            step_key = cand
            break

    # 3) Build a DataFrame with metric and (optional) step
    cols = [metric]
    if step_key is not None:
        cols.append(step_key)
    df = hist[cols]

    # keep only rows where metric is present
    df = df.dropna(subset=[metric])
    print(f"Non-NaN rows for '{metric}':", len(df))

    if df.empty:
        return None

    values = df[metric].to_numpy(dtype=float)

    if step_key is not None:
        step = df[step_key].to_numpy(dtype=float)
    else:
        step = np.arange(len(values), dtype=float)

    print(step)
    print(values)

    return step, values



def main():
    args = parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}",
                    filters={"group": args.group})

    if len(runs) == 0:
        raise RuntimeError(f"No runs found in group '{args.group}'")

    print(f"Found {len(runs)} runs in group '{args.group}'.")
    print(f"Downloading full history for metric '{args.metric}' ...")

    # Build output directory like: results/baseline/ppo_mt10
    group_path_parts = args.group.split("/")
    output_dir = os.path.join(args.output_root, *group_path_parts)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving individual run files under: {output_dir}")

    # safe_metric = sanitize_key(args.metric)
    num_saved = 0
    for idx, run in enumerate(runs):
        print(f"[{idx+1}/{len(runs)}] Run id={run.id}, name={run.name} ...", flush=True)

        # Load full history once
        samples = args.max_rows if args.max_rows is not None else 100000
        hist = run.history(samples=samples)
        if isinstance(hist, list):
            hist = pd.DataFrame(hist)

        # Find metrics that start with "charts/"
        chart_metrics = [c for c in hist.columns]
        try:
            chart_metrics.remove('charts/mean_episodic_return')
            chart_metrics.remove('charts/SPS')
        except ValueError:
            pass
        print("Found chart metrics:", chart_metrics)

        if len(chart_metrics) == 0:
            print("  -> No 'charts/*' metrics found, skipping this run.")
            continue

        # Extract step column once
        step_key = next((k for k in ["Step"] if k in hist.columns), None)
        if step_key is not None:
            step = hist[step_key].dropna().to_numpy(float)
        else:
            step = np.arange(len(hist), dtype=float)

        # Dictionary to save into .npz
        save_dict = {"step": step}

        # Fetch each charts/* metric
        for metric in chart_metrics:
            df = hist[[metric]].dropna()
            if df.empty:
                print(f"  -> Metric '{metric}' is empty, skipping.")
                continue

            values = df[metric].to_numpy(float)
            safe_key = sanitize_key(metric)
            save_dict[safe_key] = values
            print(f"  -> Added metric: {metric} ({safe_key}), {len(values)} points")

        # Save run file
        file_name = f"run_{num_saved}.npz"
        file_path = os.path.join(output_dir, file_name)
        np.savez(file_path, **save_dict)
        print(f"  -> Saved {file_path}")

        num_saved += 1

    if num_saved == 0:
        raise RuntimeError("No runs contained any non-empty charts/* metrics.")
    if num_saved == 0:
        raise RuntimeError(
            f"No runs in group '{args.group}' contained non-empty metric '{args.metric}'"
        )

    print(f"\nDone. Saved {num_saved} runs to directory: {output_dir}")


if __name__ == "__main__":
    main()



"""
Example:

python download_data.py \
  --entity whuang369-university-of-wisconsin-madison \
  --project metaworld \
  --group dro/2080/ppo_mt10/lr=0.1 \
  --metric charts/mean_success_rate

"""