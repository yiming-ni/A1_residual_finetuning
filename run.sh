run ()
{
    CUDA_VISIBLE_DEVICES=2 WANDB_LOGDIR=./wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py \
                    --env_name=A1Run-v0 \
                    --action_history=15 \
                    --utd_ratio=20 \
                    --start_training=1000 \
                    --config=configs/droq_config.py \
                    --eval_interval=4000 \
                    --object_type=$3 \
                    --object_size=$2 \
                    --residual_scale=0.1 \
                    --tqdm=False \
                    --max_steps=100000 \
                    --exp_group=t \
                    --save_video \
                    --ep_len=2000 \
                    --sparse_reward \
                    --fov \
                    --seed=$4 \
                    --ep_update=$5
}
export -f run 

parallel --delay 2 --linebuffer -j 4 run {%} {} ::: 0.12,0.12,0.12 0.1,0.1,0.1 0.08,0.08,0.08 ::: sphere box ::: 20 60 880 ::: True False
