#!/bin/bash

checkpoint_path=$HOME/Git/remdm/outputs/checkpoints/mdlm.ckpt
T=0
sampling_steps=1024 #1024
p=0.9
num_sample_batches=5 # 5000
global_batch_size=512 # useless, since we set loader.batch_size=1 and loader.eval_batch_size=1
batch_size=1        # if I try to increase this, I get OOM
eval_batch_size=1
devices=1
generated_seqs_path=$HOME/Git/remdm/outputs/mdlm_T-${sampling_steps}_topp-${p}.json
export HYDRA_FULL_ERROR=1

# hydra.run.dir="${PWD}/outputs/mdlm" \

python -u -m main \
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
    seed=1 \
    loader.batch_size=${batch_size} \
    loader.eval_batch_size=${eval_batch_size} \
    eval.perplexity_batch_size=1 \
    sampling.num_sample_batches=${num_sample_batches} \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="mdlm" \
    trainer.devices=${devices}
