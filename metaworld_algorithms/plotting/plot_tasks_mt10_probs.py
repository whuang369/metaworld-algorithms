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
        f"dro_weight_drawer_open_v3",
        f"dro_weight_drawer_close_v3",
        f"dro_weight_door_open_v3",
        f"dro_weight_push_v3",
        f"dro_weight_pick_place_v3",
        f"dro_weight_reach_v3",
        f"dro_weight_button_press_topdown_v3",
        f"dro_weight_peg_insert_side_v3",
        f"dro_weight_window_open_v3",
        f"dro_weight_window_close_v3",
    ]
    n_y_names = len(y_name_list)
    n_rows, n_cols = 2, 5

    fig = plt.figure(figsize=(5*4, 2*4 + 0.4))

    groups = [
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'lp/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0/eps_0.025',
    ]

    # ============================================================
    # Store per-task results_dicts for the normalized figure
    # ============================================================
    all_task_results: dict = {}   # y_name -> {algo: array (n_runs, n_frames)}
    shared_x = None

    algo_order = ["DRATS", "LP", "Uniform"]
    final_point = {a: [] for a in algo_order}
    final_low = {a: [] for a in algo_order}
    final_high = {a: [] for a in algo_order}
    task_names = []

    subplot_idx = 1
    for y_name in y_name_list:

        data_dict = defaultdict(list)
        x = None

        for group in groups:
            results_dir = f"results_mt10_all/{group}"
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
            if 'lp' in group:
                key = 'Learning Progress'
            elif 'lr_0/' in group:
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
        ylabel = 'Mean Task Probability'

        plot_sample_efficiency_curve(
            frames=x,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=ylabel,
            labelsize='xx-large',
            ticklabelsize='xx-large',
            marker=''
        )
        ax.set_title(y_name.replace("dro_weight_", "").replace('_return', '').replace('_', '-'), fontsize="xx-large")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize('xx-large')

    plt.tight_layout(h_pad=3)
    fig.subplots_adjust(top=0.85)
    for ax in fig.axes:
        ax.set_box_aspect(1)
    first_ax = fig.axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='xx-large', ncol=3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/mt10_tasks_probs.png", dpi=200)
    # plt.show()