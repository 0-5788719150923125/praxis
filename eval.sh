#!/bin/sh

MODEL='data/praxis'
lm_eval --model hf \
    --model_args pretrained=${MODEL},cache_dir=data/praxis \
    --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,tinyMMLU,triviaqa,winogrande \
    --device cuda:1 \
    --batch_size auto