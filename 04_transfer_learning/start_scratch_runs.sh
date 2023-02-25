#!/bin/bash

declare -a SAMPLESIZE=(120 75 50 25 10 1)
declare -a SEED=(2020 1224)
declare -a TASK=('MOTOR' 'WM' 'EMOTION' 'GAMBLING' 'LANGUAGE' 'SOCIAL' 'RELATIONAL')

for this_seed in ${SEED[@]}; do
  for ss in ${SAMPLESIZE[@]}; do
    for task in ${TASK[@]}; do
      python train_nets.py --task ${task} --sample_size ${ss} --seed ${this_seed} --hyperparameter hyperparameter.yaml \
            --wandb_group transfer-learning-from-scratch
    done
  done
done