import os
from collections import defaultdict

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

    data_dict = defaultdict(list)
    n_rows = 1
    n_cols = 1
    fig = plt.figure(figsize=(4,4))
    i = 1

    groups = [
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'lp/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0.5/eps_0.025',
        'dro/rs_10000/lr_0.0005/e_16/gs_32/eta_8/lr_0/eps_0.025',
        'smt'
    ]
    metric = 'return'
    # metric = 'success_rate'
    y_name = f"charts_mean_{metric}"

    for group in groups:
        key = group
        results_dir = f"results_mt10/{group}"
        if not os.path.exists(results_dir):
            print (f'Group {group} does not exist')
            continue

        # Now we can use dot notation which is much cleaner
        x, y = get_data(results_dir, x_name='step', y_name=y_name, truncate=50)
        if len(y) > 0:
            # if len(y) < 50: continue
            key = group

            if 'lp' in group:
                key = 'LP'
            elif 'lr_0/' in group:
                key = 'Uniform'
            else:
                key = 'DRO'

            y = y[:, ::1]
            data_dict[key].extend(y)
            x = (x[0] + 1)*2e6
            print(y.shape)


    ax = plt.subplot(n_rows, n_cols, i)
    # ax.set_title(group)
    i+=1

    results_dict = {algorithm: np.array(score) for algorithm, score in data_dict.items()}

    aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
    scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=1000)

    for key, value in scores.items():
        print(key, value.shape, scores[key][-1], cis[key][:, -1])


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
        # title=f'MT10',
        labelsize='large',
        ticklabelsize='large',
        marker='',
    )
    plt.title('MT10')
    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # set fontsize of scientific notation label
    ax.xaxis.get_offset_text().set_fontsize('large')

    # Set log scale
    # plt.xscale('log')
    # plt.yscale('log')

    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.80)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=3)

    save_dir = f'figures'
    save_name = f'mt10_{metric}.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=200)

    plt.show()
