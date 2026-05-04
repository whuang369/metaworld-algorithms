# Sweep values -----------------------------------------------------
import itertools
from http.cookiejar import offset_from_tz_string

type_to_abbr = {
    'multi_head': 'mh',
    'single_head': 'sh',
    'mlp': 'mlp',
    'moore': 'moore'
}

dro_lr_list = [0.3, 0]
baseline_type_list = ["linear", "mlp", "moore"]



wandb_entity = "nicholascorrado"
wandb_project = "fast10_lr1e-3"
wandb_project = "final"
# wandb_project = "mt10_lr5e-4"
# wandb_project = "ttest"

# Base script ------------------------------------------------------

# dro_lr_list = [3]
# dro_min_prob_list = [0.05]
# lr_list = [3e-4]
# types_list = [
#     ('multi_head', 'multi_head'),
#     # ('multi_head', 'linear')
# ]
# norm_adv_list = [1]
# rollout_steps_list = [10000]
# offset_list = [0]



base_cmd = "python dro_examples/dro_ppo_mt10.py --track"

rs = 10000
lr = 5e-4
epoch = 16
dro_lr = 0.5
dro_eta = 8
dro_eps = 0.1/4

for task_sampling_algo in ['smt']:
# for task_sampling_algo in ['dro', 'uniform']:
#     group = f"{task_sampling_algo}/soft_modules"
    # group = f"{task_sampling_algo}/pcgrad"
    # group = f"{task_sampling_algo}/paco"
    group = f"{task_sampling_algo}/B1_85"
    # group = f"{task_sampling_algo}"

    offset = 0
    name  = group
    cmd = (
        f"{base_cmd} "
        f" --env_id MT10"
        f" --seed_offset {offset}"
        f" --wandb_entity {wandb_entity}"
        f" --wandb_project {wandb_project}"
        f" --wandb_group {group}"
        f" --wandb_name {name}"
        f" --total_steps {int(100e6)}"
        f" --evaluation_frequency {2_000_000 // 500}"
        f" --learning-rate {lr}"
        f" --num_epochs {epoch}"
        f" --rollout_steps {rs}"
        f" --dro_rollout_steps {rs}"

        f" --task_sampling_algo {task_sampling_algo}"
        f" --dro_eta {dro_eta}"
        f" --dro_learning_rate {dro_lr}"
        f" --dro_eps {dro_eps}"
        # f" --value_optimizer pcgrad"
        # f" --actor_type soft_modules"
        # f" --value_type soft_modules"
    )

    print(cmd)
