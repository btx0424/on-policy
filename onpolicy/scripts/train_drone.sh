#!/bin/sh
env="drone"
scenario="navigation"  # simple_speaker_listener # simple_reference
num_agents=3
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_drone.py \
        --use_popart --env_name ${env} --algorithm_name ${algo} \
        --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
        --hidden_size 128 --layer_N 2 \
        --seed ${seed} --n_training_threads 1 \
        --n_rollout_threads 1 --num_mini_batch 1 --episode_length 1200 \
        --num_env_steps 50000000 --ppo_epoch 10 --gain 0.01 --lr 5e-4 \
        --critic_lr 5e-4 --eps_start 0 --save_interval 100 \
        --use_eval --eval_interval 20 --eval_episodes 10 --n_eval_rollout_threads 5 --eval_record
done