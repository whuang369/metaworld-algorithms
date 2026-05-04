# Sweep values -----------------------------------------------------
import itertools

type_to_abbr = {
    'multi_head': 'mh',
    'single_head': 'sh',
    'mlp': 'mlp',
    'moore': 'moore'
}

baseline_type_list = ["linear", "mlp", "moore"]

wandb_entity = "whuang369-university-of-wisconsin-madison"
wandb_project = "pointmaze"

num_tasks = 3

# Base script ------------------------------------------------------

n = 2 # Reduce HYP rate

dro_lr_list = [0, 1.0]
dro_min_prob_list = [0.167]
lr_list = [3e-4]
types_list = [
    ('multi_head', 'multi_head'),
    # ('multi_head', 'linear')
]
norm_adv_list = [1]
rollout_steps_list = [8000 // n]
reward_type_list = [1] # 0: Dense, 1: Sparse

script_path = "pointmaze/dro_ppo_pointmaze"
base_cmd = f"python dro_examples/{script_path}.py --track"
env_id = "dro_pointmaze"
episode_length = 400

for lr, types, norm_adv, dro_lr, dro_min_prob, rollout_steps, reward_type in itertools.product(
    lr_list,
    types_list,
    norm_adv_list,
    dro_lr_list,
    dro_min_prob_list,
    rollout_steps_list,
    reward_type_list,
):
    actor_type, value_type = types
    if reward_type == 0:
        str_reward_type = "Dense"
    else:
        str_reward_type = "Sparse"
    # group = f"dro/{type_to_abbr[actor_type]}_{type_to_abbr[value_type]}/norm_{norm_adv}/lr_{lr}/mp_{min_prob}"
    group = f"dro/lr_{lr}/norm_{norm_adv}/dro_lr_{dro_lr}/mp_{dro_min_prob}/{str_reward_type}/reduce_factor{n}"

    name  = group
    cmd = (
        f"{base_cmd} "
        f" --env_id {env_id}"
        f" --seed_offset {3} "
        f" --wandb_entity {wandb_entity} "
        f" --wandb_project {wandb_project} "
        f" --wandb_group {group} "
        f" --wandb_name {name} "
        f" --total_steps {int(10e6) // n}"
        # f" --evaluation_frequency {1}"
        f" --evaluation_frequency {500_000 // (episode_length * n)}"
        f" --learning_rate {lr}"
        f" --actor_type {actor_type} "
        f" --value_type {value_type} "
        f" --normalize_advantages {norm_adv}"
        f" --dro_learning_rate {dro_lr} "
        f" --dro_min_prob {dro_min_prob}"
        f" --rollout_steps {rollout_steps}"
        f" --sparse {reward_type}"
        f" --dro_eps 0.01"
    )

    print(cmd)