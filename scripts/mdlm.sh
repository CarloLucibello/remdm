#!/bin/bash
#SBATCH --job-name="mdlm"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
# #SBATCH --qos=debug # only for debug_gpu and debug_cpu partitions
#SBATCH --nodes=1
#SBATCH --gpus=1               # num gpus. If set to 0 change the partition to defq or compute
#SBATCH --cpus-per-task=8      # number of threads per task
#SBATCH --ntasks=1             # SET EQUAL TO gpus IF DOING pytorch's DDP
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mem-per-cpu=16000M  # memory per cpu core, default 8GB

#SBATCH --account=lucibello 
#SBATCH --mail-type=NONE   #notify for NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=carlo.lucibello@unibocconi.it

## AVAILABLE PARTITIONS
# defq, timelimit 3 days, Nodes=cnode0[1-4] (CPU)
# compute, timelimit 15 days, Nodes=cnode0[5-8] (CPU)
# long_gpu timelimit 3 days, Nodes=gnode0[1-2] (GPU)
# gpu timelimit 1 day, Nodes=gnode0[1-4] (GPU)
# medium_gpu, timelimit 3 hours, Nodes=gnode0[1-4] (GPU)
# stata, timelimit 3 days, Nodes=cnode08 (CPU)
# debug_cpu, timelimit 15 minutes, Nodes=cnode01 (short test on CPU)
# debug_gpu, timelimit 15 minutes, Nodes=gnode0[1-4] (short test on GPU)

## COMMON SLURM COMMANDS 
# squeue 
# sbatch job.sh
# sinfo -Nel
# scancel <job_id>

module load miniconda3
eval "$(conda shell.bash hook)"
module load cuda/12.8

conda activate remdm

start_time=$(date +%s)

SEED=1
checkpoint_path=$HOME/Git/remdm/outputs/checkpoints/mdlm.ckpt
T=0
sampling_steps=128 #1024
p=0.9
num_sample_batches=5 # 5000
global_batch_size=512 # useless, since we set loader.batch_size=1 and loader.eval_batch_size=1
batch_size=1        # if I try to increase this, I get OOM
generated_seqs_path=$HOME/Git/remdm/outputs/mdlm_T-${sampling_steps}_topp-${p}_n-${num_sample_batches}.json

export HYDRA_FULL_ERROR=1

# hydra.run.dir="${PWD}/outputs/mdlm" \

srun python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=$HOME/Git/remdm/outputs/data \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    T=${T} \
    loader.global_batch_size=${global_batch_size} \
    sampling.steps=${sampling_steps} \
    seed=${SEED} \
    loader.batch_size=${batch_size} \
    loader.eval_batch_size=${batch_size} \
    eval.perplexity_batch_size=${batch_size} \
    sampling.num_sample_batches=${num_sample_batches} \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="mdlm"


end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_hours=$(awk "BEGIN {printf \"%.4f\", $elapsed_seconds/3600}")
echo "Elapsed time: $elapsed_hours hours"
