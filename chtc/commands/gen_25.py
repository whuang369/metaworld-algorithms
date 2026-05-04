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
wandb_project = "mt25"

# Base script ------------------------------------------------------

dro_lr_list = [0, 1.0]
dro_min_prob_list = [1/25/2]
lr_list = [3e-4]
types_list = [
    ('multi_head', 'multi_head'),
    # ('multi_head', 'linear')
]
norm_adv_list = [1]

base_cmd = "python dro_examples/dro_ppo_mt10.py --track"

for lr, types, norm_adv, dro_lr, dro_min_prob in itertools.product(
    lr_list,
    types_list,
    norm_adv_list,
    dro_lr_list,
    dro_min_prob_list,
):
    actor_type, value_type = types
    # group = f"dro/{type_to_abbr[actor_type]}_{type_to_abbr[value_type]}/norm_{norm_adv}/lr_{lr}/mp_{min_prob}"
    group = f"dro/lr_{lr}/norm_{norm_adv}/lr_{dro_lr}/mp_{dro_min_prob}"

    name  = group
    cmd = (
        f"{base_cmd} "
        f" --env_id MT25"
        f" --seed_offset {0} "
        f" --wandb_entity {wandb_entity} "
        f" --wandb_project {wandb_project} "
        f" --wandb_group {group} "
        f" --wandb_name {name} "
        f" --learning_rate {lr}"
        f" --actor_type {actor_type} "
        f" --value_type {value_type} "
        f" --normalize_advantages {norm_adv}"
        f" --dro_learning_rate {dro_lr} "
        f" --dro_min_prob {dro_min_prob}"
        f" --dro_rollout_steps 10000"
        f" --dro_eps {1/25/2}"
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