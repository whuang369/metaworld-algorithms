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

rs_list = [10000]
lr_list = [5e-4]
epoch_list = [16]
gs_list = [32]
dro_lr_list = [0.5]
dro_eta_list = [8]
dro_eps_list = [0.025]

for rs, lr, ep, gs, dro_eta, dro_lr, dro_eps in itertools.product(
    rs_list,
    lr_list,
    epoch_list,
    gs_list,
    dro_eta_list,
    dro_lr_list,
    dro_eps_list,
):

    # actor_type, value_type = types

    # group = f"dro/{type_to_abbr[actor_type]}_{type_to_abbr[value_type]}/norm_{norm_adv}/lr_{lr}/mp_{min_prob}"
    # group = f"dro/lr_{lr}/rs_{rs}/norm_{norm_adv}/lr_{dro_lr}/mp_{dro_min_prob}"
    group = (f"lp/rs_{rs}/lr_{lr}/e_{ep}/gs_{gs}/eta_{dro_eta}/lr_{dro_lr}/eps_{dro_eps}")
    group = (f"uniform/moore")

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
        f" --rollout_steps {rs}"
        f" --learning_rate {lr}"
        f" --num_epochs {ep}"
        f" --num_gradient_steps {gs}"
        f" --dro_rollout_steps {rs}"
        f" --dro_eta {dro_eta}"
        f" --dro_learning_rate {dro_lr}"
        f" --dro_eps {dro_eps}"

        # f" --dro_eps 0.05"
        # f" --dro_min_prob {dro_min_prob}"
        # f" --actor_type {actor_type} "
        # f" --value_type {value_type} "
        # f" --dro_eps 0.05"
        # f" --dro_min_prob {dro_min_prob}"
    )

    print(cmd)
