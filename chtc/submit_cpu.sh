results_dir=results/${1}
log_dir=logs/${1}
commands_file=scripts_cpu/${1}

mkdir -p ${results_dir}
mkdir -p ${log_dir}

condor_submit job_cpu.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir} \
  commands_file=${commands_file} \
  num_jobs=${2:-1}