import os
import warnings

import numpy as np

def get_data(results_dir, x_name='timestep', y_name='returns', lable_name='env_ids', id_name='task_ids', filename='stats.npz'):

    print (results_dir)
    paths = []
    try:
        for subdir in os.listdir(results_dir):
            if 'run_' in subdir:
                cur_path = f'{results_dir}/{subdir}/{filename}'
                if os.path.isfile(cur_path):
                    paths.append(cur_path)
                else:
                    print (f'No file at {cur_path}!')
    except Exception as e:
        print(e)

    if len(paths) == 0:
        # warnings.warn(f'No data found at: {results_dir}')
        print(f'No data found at: {results_dir}')

        return None, None

    y_list = []

    x = None
    length = None
    ids = None

    for path in paths:
        with np.load(path, allow_pickle=True) as data_file:
            for x in data_file.files:
                print(x)
                print(data_file[x])

            if x is None: x = data_file[x_name]
            y = data_file[y_name]

            y_list.append(y)
            z = data_file[lable_name]
            ids = data_file[id_name]

    return x, np.array(y_list), z, ids

