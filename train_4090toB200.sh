#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$HOME/data/math}"
CKPT_DIR="${CKPT_DIR:-/mnt/OS-oKqEXySb/share/traces/claude/huyan/checkpoints/PPO-MATH/PPO-B200_neoSFT_MATH}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/verl/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "verl Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
mkdir -p "$CKPT_DIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.sft_loss_coeff=0.05 \
    actor_rollout_ref.actor.sft_start_step=100 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.use_v1=False \
    trainer.logger="['console','wandb']" \
    trainer.project_name=PPO-MATH \
    trainer.experiment_name=PPO-B200_neoSFT_MATH \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    "$@" 2>&1 | tee verl_demo.log
