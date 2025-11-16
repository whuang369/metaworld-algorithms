results_dir=results_a100/${1}
log_dir=logs_a100/${1}
mkdir -p ${results_dir}
mkdir -p ${log_dir}

# make a temp commands file where spaces are replaced with * so that multi-word commands are properly parsed by the queue command
commands_file=commands/${1}.sh
commands_file_tmp=commands/${1}_tmp.sh
sed 's/ /*/g' "$commands_file" > "$commands_file_tmp"

USER_NAME=whuang369

condor_submit job_a100.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir} \
  commands_file=${commands_file_tmp} \
  num_jobs=${2:-1} \
  job_length=${3:-"short"} \
  user_name=${USER_NAME}

rm $commands_file_tmp