#!/bin/bash

declare -a SAMPLESIZE=(10 25)
declare -a SEED=(2020 1224)
declare -a TASK=('MOTOR' 'WM' 'EMOTION' 'GAMBLING' 'LANGUAGE' 'SOCIAL' 'RELATIONAL')

for this_seed in ${SEED[@]}; do
  for ss in ${SAMPLESIZE[@]}; do
    for task in ${TASK[@]}; do
      python train_nets.py --task ${task} --sample_size ${ss} --seed ${this_seed}
    done
  done
done