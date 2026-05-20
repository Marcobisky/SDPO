#!/bin/bash

# Usage: ./run_local_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Default to ToolUse dataset
DATA_PATH="datasets/tooluse"

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# Set your personal wandb entity so logs go to your own account
export WANDB_ENTITY="marcobisky"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-local \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
vars.dir=$PROJECT_ROOT \
trainer.default_local_dir=$PROJECT_ROOT/checkpoints/$EXP_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
actor_rollout_ref.rollout.max_num_seqs=256"

echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

# Kill any leftover Ray workers from previous runs to free GPU memory/NCCL state
ray stop --force 2>/dev/null || true
sleep 3

# Auto-preprocess: convert JSON -> parquet if parquet files are missing
TRAIN_PARQUET="$PROJECT_ROOT/$DATA_PATH/train.parquet"
TEST_PARQUET="$PROJECT_ROOT/$DATA_PATH/test.parquet"
if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$TEST_PARQUET" ]; then
    echo "Parquet files not found. Running preprocessing..."
    python "$PROJECT_ROOT/data/preprocess.py" --data_source "$PROJECT_ROOT/$DATA_PATH"
    echo "Preprocessing complete."
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
