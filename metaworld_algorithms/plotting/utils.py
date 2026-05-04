import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rliable.plot_utils import _annotate_and_decorate_axis


def get_data(results_dir, x_name='step', y_name='charts_mean_success_rate', truncate=None):
    print(results_dir)

    # Gather all valid file paths
    paths = []
    try:
        for subdir in os.listdir(results_dir):
            if 'run_' in subdir:
                cur_path = os.path.join(results_dir, subdir)
                if os.path.isfile(cur_path):
                    paths.append(cur_path)
                else:
                    print(f'No file at {cur_path}!')
    except Exception as e:
        print(e)

    if len(paths) == 0:
        print(f'No data found at: {results_dir}')
        return None, None

    # Read all y arrays (and x if present)
    raw_runs = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data_file:
            # for metric in data_file.files:
            #     print(metric)
            if y_name not in data_file.files:
                continue

            y = data_file[y_name]
            print(path, len(y))

            # Optional truncate
            if truncate is not None:
                y = y[:truncate]

            # Load x if present, otherwise default to range
            # if x_name in data_file.files:
            #     x = data_file[x_name]
            #     if truncate is not None:
            #         x = x[:truncate]
            # else:
            #     x = np.arange(len(y))

            x = np.arange(len(y))

            raw_runs.append((x, y))

    if len(raw_runs) == 0:
        print("No usable runs found.")
        return None, None

    # Determine max length
    lengths = [len(y) for (_, y) in raw_runs]
    max_len = max(lengths)

    # Filter runs that match the max length
    filtered = [(x, y) for (x, y) in raw_runs if len(y) == max_len]

    if len(filtered) == 0:
        print("No runs match the maximum length.")
        return None, None

    # Build X and Y matrices
    X = np.stack([x[3:] for (x, _) in filtered], axis=0)  # shape (num_runs, max_len)
    Y = np.stack([y[3:] for (_, y) in filtered], axis=0)  # shape (num_runs, max_len)

    return X, Y


def plot_sample_efficiency_curve_with_error_bars(frames,
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
        frames,
        metric_values,
        color=colors[algorithm],
        marker=kwargs.get('marker', 'o'),
        linewidth=kwargs.get('linewidth', 2),
        label=algorithm)
    mid = (upper + lower) / 2
    err = (upper - lower) / 2
    ax.errorbar(frames, mid, yerr=err, color=colors[algorithm], fmt='none', capsize=4, capthick=1.5, elinewidth=1.5)
  kwargs.pop('marker', '0')
  kwargs.pop('linewidth', '2')

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)
