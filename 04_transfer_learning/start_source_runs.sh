#!/bin/bash

declare -a SEED=(7474)
declare -a TASK=('MOTOR' 'WM' 'EMOTION' 'GAMBLING' 'LANGUAGE' 'SOCIAL' 'RELATIONAL')

for this_seed in ${SEED[@]}; do
  for task in ${TASK[@]}; do
    python train_source_nets.py --left_out_task ${task} --seed ${this_seed} --hyperparameter hyperparameter.yaml \
          --job_type source-nets2 --wandb_group transfer-learning-source-nets
  done
done