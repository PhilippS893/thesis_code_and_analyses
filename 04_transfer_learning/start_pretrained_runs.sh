#!/bin/bash

#declare -a SAMPLESIZE=(120 75 50 25 10 1)
declare -a SAMPLESIZE=(1)
declare -a SEED=(17 1337)
declare -a TASK=('MOTOR' 'WM' 'EMOTION' 'LANGUAGE' 'SOCIAL' 'RELATIONAL' 'GAMBLING')
declare -a BESTFOLDS=(3 8 6 6 5 3 7)
#declare -a TASK=('WM' 'MOTOR' 'LANGUAGE' 'EMOTION' 'GAMBLING' 'RELATIONAL' 'SOCIAL')
#declare -a BESTFOLDS=(0 9 9 1 3 7 3)

for this_seed in ${SEED[@]}; do
  for ss in ${SAMPLESIZE[@]}; do
    for i in ${!TASK[@]}; do
      #python train_nets.py --task ${TASK[i]} --sample_size ${ss} --seed ${this_seed} --transfer_learning True \
      #      --best_fold ${BESTFOLDS[i]} --wandb_group transfer-learning-pretrained
      python train_nets.py --task ${TASK[i]} --sample_size ${ss} --seed ${this_seed} --transfer_learning True \
            --best_fold ${BESTFOLDS[i]} --hyperparameter hyperparameter_old.yaml --wandb_group transfer-learning
    done
  done
done