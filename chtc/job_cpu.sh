#!/bin/bash

CODENAME=metaworld-algorithms
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz
cd ${CODENAME}

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[cpu]"
uv pip install -e ./metaworld
uv pip install -e ./metaworld/custom-envs
uv pip install packaging

wandb login 7313077863c8908c24cc6058b99c2b2cc35d326b
export WANDB_MODE=offline

pid=$1  # ranges from 0 to num_commands*num_jobs-1
step=$2 # ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
cmd="${cmd} --seed ${step}"
echo $cmd

tar czvf results_${pid}.tar.gz run_results
mv results_${pid}.tar.gz ..

$cmd