#!/bin/bash

CODENAME=metaworld-algorithms
USER_NAME=$4
cp /staging/${USER_NAME}/${CODENAME}.tar.gz .
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

wandb login d0dbec0b8f79cdb57ef36ae46bb16a336954b2ca

pid=$1  # ranges from 0 to num_commands*num_jobs-1
step=$2 # ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
cmd="${cmd} --seed ${step}"
echo $cmd

#tar czvf results_${pid}.tar.gz run_results
#mv results_${pid}.tar.gz ..

$cmd