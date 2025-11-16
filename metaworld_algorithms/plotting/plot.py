import os

import numpy as np
import seaborn

import matplotlib
# from sympy.printing.pretty.pretty_symbology import line_width

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data

if __name__ == "__main__":

    data_dict = {}
    seaborn.set_theme(style='whitegrid')

    n_rows = 1
    n_cols = 3
    fig = plt.figure(figsize=(9,3))
    i = 1

    groups = [
        'baseline_gpu_ppo_mt10',
        'dro_ppo_mt10_lr=0.1_gpu',
        'dro_gpu_ppo_mt10_lr=0.3',
        # 'dro_gpu_ppo_mt10_lr=0.3',
    ]

    for group in groups:
        key = f"PPO"
        results_dir = f"wandb_export/metaworld_dro/{group}"
        if not os.path.exists(results_dir):
            print (f'Group {group} does not exist')
            continue

        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_title(group)
        i+=1

        # Now we can use dot notation which is much cleaner
        x, y = get_data(results_dir, x_name='timestep', y_name='charts/mean_success_rate', filename='stats.npz')
        if y is not None:
            data_dict[key] = y

    results_dict = {algorithm: score for algorithm, score in data_dict.items()}
    aggr_func = lambda scores: np.array([metrics.aggregate_iqm([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

    plot_sample_efficiency_curve(
        frames=x,
        point_estimates=scores,
        interval_estimates=cis,
        ax=ax,
        algorithms=None,
        xlabel='Timestep',
        ylabel=f'Return',
        # title=f'{env_id}',
        labelsize='large',
        ticklabelsize='large',
    )
    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # set fontsize of scientific notation label
    ax.xaxis.get_offset_text().set_fontsize('large')

    # Set log scale
    # plt.xscale('log')
    # plt.yscale('log')

    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.88)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large')

    save_dir = f'figures'
    save_name = f'return.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
