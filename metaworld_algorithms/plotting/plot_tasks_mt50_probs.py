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

from utils import get_data, plot_sample_efficiency_curve_with_error_bars

if __name__ == "__main__":

    metric = "return"
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

    fig = plt.figure(figsize=(5 * 3, 10 * 3))
    groups = [
        # 'dro/rs_10000/lr_0.0005/ep_16/eta_3/lr_0.5/e_0.005',
        # 'dro/rs_10000/lr_0.0005/ep_16/eta_3/lr_0/e_0.005',
        'dro',
        'learning_progress',
        'uniform',
    ]

    # ============================================================
    # Store per-task results_dicts for the normalized figure
    # ============================================================
    all_task_results: dict = {}   # y_name -> {algo: array (n_runs, n_frames)}
    shared_x = None

    algo_order = ["DRATS", "Learning Progress", "Uniform"]
    final_point = {a: [] for a in algo_order}
    final_low = {a: [] for a in algo_order}
    final_high = {a: [] for a in algo_order}
    task_names = []

    subplot_idx = 1
    for y_name in y_name_list:

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
                x = x_vals
                x = (x[0] + 1) * 2e6

            key = group
            if 'learning_progress' in group:
                key = 'Learning Progress'
            elif 'uniform' in group:
                key = 'Uniform'
            else:
                key = 'DRATS'

            data_dict[key].extend(y_vals)

        if len(data_dict) == 0:
            print(f"WARNING: No data found for metric {y_name}, skipping.")
            continue

        results_dict = {algo: np.array(scores) for algo, scores in data_dict.items()}

        # Save for normalized figure
        all_task_results[y_name] = results_dict
        if shared_x is None and x is not None:
            shared_x = x

        aggr_func = lambda scores: np.array(
            [metrics.aggregate_mean([scores[..., frame]])
             for frame in range(scores.shape[-1])]
        )
        print(y_name)

        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

        ax = plt.subplot(n_rows, n_cols, subplot_idx)
        subplot_idx += 1
        ylabel = 'Mean Return' if 'return' in y_name else 'Mean Success Rate'

        plot_sample_efficiency_curve(
            frames=x,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=ylabel[:-7],
            labelsize='medium',
            ticklabelsize='medium',
            marker=''
        )
        ax.set_title(y_name.replace("charts_", ""), fontsize="large")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize('medium')

    plt.tight_layout()
    fig.subplots_adjust(top=0.98)
    first_ax = fig.axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large', ncol=3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/mt50_tasks_{metric}.png", dpi=200)
    # plt.show()

    # ============================================================
    # 🔥 Figure 2: Average Normalised Return across all tasks
    # ============================================================
    # For each task, find the max return observed by any algorithm at any
    # run/frame, then divide every algorithm's array by that value.
    # Finally average the normalised arrays across tasks per algorithm.

    # Collect all algorithm names present across tasks
    all_algos = set()
    for rd in all_task_results.values():
        all_algos.update(rd.keys())

    # normalized_per_task[algo] = list of (n_runs, n_frames) arrays, one per task
    normalized_per_task: dict = defaultdict(list)

    for y_name, results_dict in all_task_results.items():
        # Max return for this task across all algorithms
        task_max = max(arr.max() for arr in results_dict.values())
        if task_max == 0:
            print(f"WARNING: task_max is 0 for {y_name}, skipping normalisation.")
            continue

        for algo, arr in results_dict.items():
            normalized_per_task[algo].append(arr / task_max)

    # Average normalised arrays across tasks → (n_runs, n_frames) per algo
    # Tasks may have different run-counts; use the minimum for a clean stack,
    # or simply average along the task axis after aligning on runs.
    avg_norm_results: dict = {}
    for algo, task_arrays in normalized_per_task.items():
        min_runs = min(a.shape[0] for a in task_arrays)
        stacked = np.stack([a[:min_runs] for a in task_arrays], axis=0)  # (n_tasks, n_runs, n_frames)
        avg_norm_results[algo] = stacked.mean(axis=0)                    # (n_runs, n_frames)

    aggr_func = lambda scores: np.array(
        [metrics.aggregate_mean([scores[..., frame]])
         for frame in range(scores.shape[-1])]
    )

    scores_norm, cis_norm = rly.get_interval_estimates(avg_norm_results, aggr_func, reps=1000)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    plot_sample_efficiency_curve(
        frames=shared_x,
        point_estimates=scores_norm,
        interval_estimates=cis_norm,
        ax=ax2,
        algorithms=None,
        xlabel='Timestep',
        ylabel='Mean Aggregate Return\n(Normalized)',
        labelsize='x-large',
        ticklabelsize='x-large',
        marker=''
    )
    # ax2.set_title("Average Normalied Return", fontsize="large")
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.xaxis.get_offset_text().set_fontsize('x-large')

    handles2, labels2 = ax2.get_legend_handles_labels()
    # fig2.legend(handles2, labels2, loc='upper center', fontsize='large', ncol=len(avg_norm_results))
    fig2.tight_layout()
    ax2.legend(handles2, labels2, loc='lower right', fontsize='large', ncol=1)
    # fig2.subplots_adjust(top=0.80)
    ax2.set_box_aspect(1)
    plt.savefig(f"figures/mt50_avg_norm_{metric}.png", dpi=200)
    # plt.show()

    # ============================================================
    # Figure 3: Normalised final-performance sorted plot
    # ============================================================
    algos = list(normalized_per_task.keys())

    # (n_runs, n_tasks) — final frame of each task's normalised array
    final_norm_results = {
        a: np.stack([arr[:, -5:].mean(axis=-1) for arr in normalized_per_task[a]], axis=-1)
        for a in algos
    }

    aggr_func_tasks = lambda scores: np.array([
        metrics.aggregate_mean([scores[..., t]])
        for t in range(scores.shape[-1])
    ])
    scores_final, cis_final = rly.get_interval_estimates(final_norm_results, aggr_func_tasks, reps=1000)

    sorted_point, sorted_cis = {}, {}
    for a in algos:
        order = np.argsort(scores_final[a])[::-1]
        sorted_point[a] = scores_final[a][order]
        sorted_cis[a] = cis_final[a][:, order]

    rank_x = np.arange(1, len(normalized_per_task[algos[0]]) + 1, dtype=np.float64)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    plot_sample_efficiency_curve_with_error_bars(
        frames=rank_x,
        point_estimates=sorted_point,
        interval_estimates=sorted_cis,
        ax=ax3,
        algorithms=None,
        xlabel='Task Rank (best → worst)',
        ylabel='Final Mean Return (Normalized)',
        labelsize='x-large',
        ticklabelsize='x-large',
        marker='o',
    )
    # ax3.set_xticks(rank_x)
    # ax3.set_title('', fontsize='large')
    # ax3.set_yscale('log')
    handles3, labels3 = ax3.get_legend_handles_labels()
    # fig3.legend(handles3, labels3, loc='upper center', fontsize='large', ncol=len(algos))
    ax3.legend(handles3, labels3, loc='lower left', fontsize='large')
    fig3.tight_layout()
    ax3.set_box_aspect(0.5)
    # fig3.subplots_adjust(top=0.80)
    plt.savefig(f"figures/mt50_tasks_{metric}_final_norm_sorted.png", dpi=200)
    # plt.show()