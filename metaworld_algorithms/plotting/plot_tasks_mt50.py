import os
from collections import defaultdict

import numpy as np
import seaborn
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    # seaborn.set_theme(style='whitegrid')

    metric = "return"
    # metric = "success_rate"
    # ============================================================
    # 🔥 LIST OF METRICS YOU WANT TO PLOT
    # ============================================================
    y_name_list = [
        f"charts_assembly_v3_{metric}",
        f"charts_basketball_v3_{metric}",
        f"charts_bin_picking_v3_{metric}",
        f"charts_box_close_v3_{metric}",
        f"charts_button_press_topdown_v3_{metric}",
        f"charts_button_press_topdown_wall_v3_{metric}",
        f"charts_button_press_v3_{metric}",
        f"charts_button_press_wall_v3_{metric}",
        f"charts_coffee_button_v3_{metric}",
        f"charts_coffee_pull_v3_{metric}",
        f"charts_coffee_push_v3_{metric}",
        f"charts_dial_turn_v3_{metric}",
        f"charts_disassemble_v3_{metric}",
        f"charts_door_close_v3_{metric}",
        f"charts_door_lock_v3_{metric}",
        f"charts_door_open_v3_{metric}",
        f"charts_door_unlock_v3_{metric}",
        f"charts_hand_insert_v3_{metric}",
        f"charts_drawer_close_v3_{metric}",
        f"charts_drawer_open_v3_{metric}",
        f"charts_faucet_open_v3_{metric}",
        f"charts_faucet_close_v3_{metric}",
        f"charts_hammer_v3_{metric}",
        f"charts_handle_press_side_v3_{metric}",
        f"charts_handle_press_v3_{metric}",
        f"charts_handle_pull_side_v3_{metric}",
        f"charts_handle_pull_v3_{metric}",
        f"charts_lever_pull_v3_{metric}",
        f"charts_peg_insert_side_v3_{metric}",
        f"charts_pick_place_wall_v3_{metric}",
        f"charts_pick_out_of_hole_v3_{metric}",
        f"charts_reach_v3_{metric}",
        f"charts_push_back_v3_{metric}",
        f"charts_push_v3_{metric}",
        f"charts_pick_place_v3_{metric}",
        f"charts_plate_slide_v3_{metric}",
        f"charts_plate_slide_side_v3_{metric}",
        f"charts_plate_slide_back_v3_{metric}",
        f"charts_plate_slide_back_side_v3_{metric}",
        f"charts_peg_unplug_side_v3_{metric}",
        f"charts_soccer_v3_{metric}",
        f"charts_stick_push_v3_{metric}",
        f"charts_stick_pull_v3_{metric}",
        f"charts_push_wall_v3_{metric}",
        f"charts_reach_wall_v3_{metric}",
        f"charts_shelf_place_v3_{metric}",
        f"charts_sweep_into_v3_{metric}",
        f"charts_sweep_v3_{metric}",
        f"charts_window_open_v3_{metric}",
        f"charts_window_close_v3_{metric}",
    ]
    # 10 metrics → 2×5 grid
    n_y_names = len(y_name_list)
    n_rows, n_cols = 10, 5

    fig = plt.figure(figsize=(5*4, 10*4))

    groups = [
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'lp/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0/eps_0.025',
    ]

    # NEW: collect final point (+ CI) per task for a sorted summary plot
    algo_order = ["DRO", "Learning Progress", "Uniform"]
    final_point = {a: [] for a in algo_order}
    final_low = {a: [] for a in algo_order}
    final_high = {a: [] for a in algo_order}
    task_names = []  # aligned with the final_* lists (only tasks where both algos exist)

    # ============================================================
    # 🔥 MAIN METRIC LOOP
    # ============================================================
    subplot_idx = 1
    for y_name in y_name_list:

        # Reset container for each metric
        data_dict = defaultdict(list)
        x = None

        for group in groups:
            results_dir = f"results_mt50/{group}"
            if not os.path.exists(results_dir):
                print(f'Group {group} does not exist')
                continue

            x_vals, y_vals = get_data(
                results_dir,
                x_name='step',
                y_name=y_name,
                truncate=50,
            )
            if y_vals is None:
                continue

            if x is None:
                x = x_vals  # keep same x for all groups
                x = np.arange(len(y_vals[0])) * 2e6

            key = group
            if 'lp' in group:
                key = 'Learning Progress'
            elif 'lr_0/' in group:
                key = 'Uniform'
            else:
                key = 'DRO'

            data_dict[key].extend(y_vals)
            # if key == 'Uniform':
            #     data_dict[key].extend(y_vals)

        # ----- Skip if nothing loaded -----
        if len(data_dict) == 0:
            print(f"WARNING: No data found for metric {y_name}, skipping.")
            continue

        # ----- Convert to arrays -----
        results_dict = {algo: np.array(scores)
                        for algo, scores in data_dict.items()}

        # ----- Compute intervals -----
        aggr_func = lambda scores: np.array(
            [metrics.aggregate_mean([scores[..., frame]])
             for frame in range(scores.shape[-1])]
        )
        print(y_name)

        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

        # NEW: stash final point (+ CI) for this task (only if both algos present)
        if all(a in scores for a in algo_order):
            task_names.append(y_name.replace("charts_", "").replace(f"_{metric}", ""))
            for a in algo_order:
                final_point[a].append(float(scores[a][-1]))
                final_low[a].append(float(cis[a][0, -1]))   # lower CI
                final_high[a].append(float(cis[a][1, -1]))  # upper CI

        # ============================================================
        # 🔥 Plot into subplot
        # ============================================================
        ax = plt.subplot(n_rows, n_cols, subplot_idx)
        subplot_idx += 1
        if 'return' in y_name:
            ylabel = 'Mean Return'
        else:
            ylabel = 'Mean Success Rate'

        plot_sample_efficiency_curve(
            frames=x,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=ylabel,
            labelsize='medium',
            ticklabelsize='medium',
        )
        # plt.ylim(0,1.05)
        ax.set_title(y_name.replace("charts_", ""), fontsize="large")

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize('medium')

    # ============================================================
    # 🔥 Global Legend & Saving
    # ============================================================
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    # global legend from first axis
    first_ax = fig.axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large', ncol=2)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/mt10_tasks_{metric}.png", dpi=200)

    # ============================================================
    # NEW: Final-performance sorted plot (worst -> best)
    # ============================================================
    # ============================================================
    # NEW: Final-performance sorted plot
    # - DRO sorted by DRO final (worst -> best)
    # - Uniform sorted by Uniform final (worst -> best)
    # - x-axis is just rank index 1..N (no task labels)
    # ============================================================
    if len(task_names) > 0:
        fig2 = plt.figure(figsize=(10, 4))
        ax2 = plt.gca()

        for a in algo_order:
            y = np.array(final_point[a], dtype=np.float64)
            lo = np.array(final_low[a], dtype=np.float64)
            hi = np.array(final_high[a], dtype=np.float64)

            order = np.argsort(y)  # sort independently per algorithm (worst -> best)
            y = y[order]
            lo = lo[order]
            hi = hi[order]

            xs = np.arange(1, len(y) + 1)  # 1..N
            ax2.plot(xs, y, marker='o', label=a)
            ax2.fill_between(xs, lo, hi, alpha=0.2)

        ax2.set_xlabel('Task rank (worst → best)')
        ax2.set_ylabel(ylabel)
        ax2.set_title('Final performance (independently sorted per algorithm)')
        ax2.set_xticks(np.arange(1, len(task_names) + 1))
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"figures/mt10_tasks_{metric}_final_sorted.png", dpi=200)

    plt.show()