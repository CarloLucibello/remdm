#!/bin/bash

checkpoint_path=../outputs/checkpoints/mdlm.ckpt
T=0
sampling_steps=1024
p=0.9
generated_seqs_path=../outputs/mdlm_T-${sampling_steps}_topp-${p}.json

export HYDRA_FULL_ERROR=1

python -u -m main \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/mdlm" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=5000 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.sampler="mdlm"