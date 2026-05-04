# Sweep values -----------------------------------------------------
import itertools

type_to_abbr = {
    'multi_head': 'mh',
    'single_head': 'sh',
    'mlp': 'mlp',
    'moore': 'moore'
}

dro_lr_list = [0.3, 0]
baseline_type_list = ["linear", "mlp", "moore"]


wandb_entity = "nicholascorrado"
wandb_project = "mt50_final"

# Base script ------------------------------------------------------


base_cmd = "python dro_examples/dro_ppo_mt10.py --track"

# rs_list = [10000]
# lr_list = [5e-4]
# epoch_list = [16]
# dro_lr_list = [0.5]
# dro_eta_list = [8]
# dro_eps_list = [0.02/4]
# for rs, lr, ep, dro_eta, dro_lr, dro_eps in itertools.product(
#     rs_list,
#     lr_list,
#     epoch_list,
#     dro_eta_list,
#     dro_lr_list,
#     dro_eps_list,
# ):

rs = 10000
lr = 5e-4
epoch = 16
dro_lr = 0.5
dro_eta = 3
dro_eps = 0.02/4

task_sampling_algos = ["dro"]

# for task_sampling_algo in itertools.product(
#     task_sampling_algos,
# ):

# for task_sampling_algo in ['dro', 'learning_progress']:
for task_sampling_algo in ['dro', 'uniform']:

    group = f"moore_both/{task_sampling_algo}/dro_eta_{dro_eta}"

    name = group
    cmd = (
        f"{base_cmd} "
        f" --env_id MT50"
        # f" --seed_offset {seed}"
        f" --wandb_entity {wandb_entity}"
        f" --wandb_project {wandb_project}"
        f" --wandb_group {group}"
        f" --wandb_name {name}"
        f" --total_steps {int(500e6)}"
        f" --evaluation_frequency {10_000_000 // 500}"
        f" --entropy-coefficient {0.03}"
        f" --learning-rate {lr}"
        f" --num_epochs {epoch}"
        f" --rollout_steps {rs}"
        f" --dro_rollout_steps {rs}"

        f" --task_sampling_algo {task_sampling_algo}"
        f" --dro_eta {dro_eta}"
        f" --dro_learning_rate {dro_lr}"
        f" --dro_eps {dro_eps}"
        f" --actor_type moore"
        f" --value_type moore"
    )

    print(cmd)


# actor_type_list = ["moore"]
# baseline_type_list = ["moore"]
#
# for lr in itertools.product(
#     dro_lr_list,
#     []
# ):
#
#     group = f"dro/moore/lr_{lr}"
#     name  = group
#
#     cmd = (
#         f"{base_cmd} "
#         f"--wandb_entity {wandb_entity} "
#         f"--wandb_project {wandb_project} "
#         f"--wandb_group {group} "
#         f"--wandb_name {name} "
#         f"--dro_learning_rate {lr} "
#         f"--baseline_type moore "
#         f"--actor_type moore "
#         f"--seed_offset {5}"
#     )
#
#     print(cmd)