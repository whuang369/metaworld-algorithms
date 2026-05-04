import os
from collections import defaultdict

import numpy as np
import seaborn
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve, _annotate_and_decorate_axis

from utils import get_data, plot_sample_efficiency_curve_with_error_bars

def plot_sample_efficiency_curve(frames,
                                 point_estimates,
                                 interval_estimates,
                                 algorithms=None,
                                 colors=None,
                                 color_palette='colorblind',
                                 figsize=(7, 5),
                                 xlabel=r'Number of Frames (in millions)',
                                 ylabel='Aggregate Human Normalized Score',
                                 ax=None,
                                 labelsize='xx-large',
                                 ticklabelsize='xx-large',
                                 **kwargs):
  """Plots an aggregate metric with CIs as a function of environment frames.

  Args:
    frames: Array or list containing environment frames to mark on the x-axis.
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metric corresponding to the values in `frames`.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Dictionary that maps each algorithm to a color. If None, then this
      mapping is created based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    ax: `matplotlib.axes` object.
    labelsize: Font size of the x-axis label.
    ticklabelsize: Font size of the ticks.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    `axes.Axes` object containing the plot.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)
  if algorithms is None:
    algorithms = list(point_estimates.keys())
  if colors is None:
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))

  for algorithm in algorithms:
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ax.plot(
        frames[algorithm],
        metric_values,
        color=colors[algorithm],
        marker=kwargs.get('marker', 'o'),
        linewidth=kwargs.get('linewidth', 2),
        label=algorithm)
    ax.fill_between(
        frames[algorithm], y1=lower, y2=upper, color=colors[algorithm], alpha=0.2)
  kwargs.pop('marker', '0')
  kwargs.pop('linewidth', '2')

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)


if __name__ == "__main__":

    metric = "return"
    y_name_list = [
        f"charts_drawer_open_v3_{metric}",
        f"charts_drawer_close_v3_{metric}",
        f"charts_door_open_v3_{metric}",
        f"charts_push_v3_{metric}",
        f"charts_pick_place_v3_{metric}",
        f"charts_reach_v3_{metric}",
        f"charts_button_press_topdown_v3_{metric}",
        f"charts_peg_insert_side_v3_{metric}",
        f"charts_window_open_v3_{metric}",
        f"charts_window_close_v3_{metric}",
    ]
    n_y_names = len(y_name_list)
    n_rows, n_cols = 2, 5

    fig = plt.figure(figsize=(5*4, 2*4 + 0.4))

    groups = [
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'smt',
        'learning_potential',
        'lp/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0/eps_0.025',
    ]

    # ============================================================
    # Store per-task results_dicts for the normalized figure
    # ============================================================
    all_task_results: dict = {}   # y_name -> {algo: array (n_runs, n_frames)}
    shared_x = None
    x_dict = {}

    algo_order = ["DRATS", "Hard First", "Learning Potential", "Learning Progress", "Uniform"]
    final_point = {a: [] for a in algo_order}
    final_low = {a: [] for a in algo_order}
    final_high = {a: [] for a in algo_order}
    task_names = []

    subplot_idx = 1
    for y_name in y_name_list:

        data_dict = defaultdict(list)
        x = None

        for group in groups:
            results_dir = f"results_mt10/{group}"
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

            x = x_vals
            x = (x[0] + 1) * 2e6


            key = group
            if 'lp' in group:
                key = 'Learning Progress'
            elif 'lr_0/' in group:
                key = 'Uniform'
            elif 'smt' in group:
                key = 'Hard First'
            elif 'learning_potential' in group:
                key = 'Learning Potential'
            else:
                key = 'DRATS'

            data_dict[key].extend(y_vals)
            x_dict[key] = x

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

        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

        ax = plt.subplot(n_rows, n_cols, subplot_idx)
        subplot_idx += 1
        ylabel = 'Mean Return' if 'return' in y_name else 'Mean Success Rate'

        plot_sample_efficiency_curve(
            frames=x_dict,
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
        ax.set_title(y_name.replace("charts_", "").replace('_return', '').replace('_', '-'), fontsize="xx-large")
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
    plt.savefig(f"figures/mt10_tasks_{metric}.png", dpi=200)
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

    scores_norm, cis_norm = rly.get_interval_estimates(avg_norm_results, aggr_func, reps=100)

    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    plot_sample_efficiency_curve(
        frames=x_dict,
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
    # ax2.set_title("Average Normalised Return (all tasks)", fontsize="large")
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.xaxis.get_offset_text().set_fontsize('x-large')

    handles2, labels2 = ax2.get_legend_handles_labels()
    # fig2.legend(handles2, labels2, loc='upper center', fontsize='large', ncol=len(avg_norm_results))
    fig2.tight_layout()
    ax2.legend(handles2, labels2, loc='lower right', fontsize='large', ncol=1)
    # fig2.subplots_adjust(top=0.85)
    ax2.set_box_aspect(1)
    plt.savefig(f"figures/mt10_avg_norm_{metric}.png", dpi=200)
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

    fig3, ax3 = plt.subplots(figsize=(4.5, 4))
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
    ax3.set_xticks(rank_x)
    # ax3.set_yscale('log')
    # ax3.set_title('Final Normalized Return', fontsize='large')
    handles3, labels3 = ax3.get_legend_handles_labels()
    # fig3.legend(handles3, labels3, loc='upper center', fontsize='large', ncol=len(algos))
    ax3.legend(handles3, labels3, loc='lower left', fontsize='large')
    fig3.tight_layout()
    ax3.set_box_aspect(1)
    # fig3.subplots_adjust(top=0.80)
    plt.savefig(f"figures/mt10_tasks_{metric}_final_norm_sorted.png", dpi=200)
    # plt.show()