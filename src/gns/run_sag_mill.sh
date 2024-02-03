#!/bin/bash
MASTER_PORT="29500"
MASTER_ADDR="localhost"

RESULT_DIR="./results"
CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"

DATASET_NAME="SAGMill"
NTRAINING_STEPS=60000
NVALID_STEPS=1000

DATA_PATH="/mnt/raid0sata1/hcj/particle/SAG_Mill_Export_Data"
MODEL_PATH="${RESULT_DIR}/models/${DATASET_NAME}/"
ROLLOUT_PATH="${RESULT_DIR}/rollouts/${DATASET_NAME}/"

# if [ $1 = "train" ]; then
#     echo "Training"
# fi
# val=`expr $NTRAINING_STEPS / $NVALID_STEPS`
# val=`expr $val \* $NVALID_STEPS`

if [ $1 = "train" ]; then
    echo "Training"
    # python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="latest" --train_state_file="latest" --ntraining_steps=${NTRAINING_STEPS} --nvalid_steps=${NVALID_STEPS} --mode='train' --exp_id=$2
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    ./gns/train.py \
    --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="latest" \
    --train_state_file="latest" --ntraining_steps=${NTRAINING_STEPS} \
    --nvalid_steps=${NVALID_STEPS} --mode='train' --exp_id=$2

elif [ $1 = "valid" ]; then
    echo "Validating"
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="$2" --mode="valid"  --cuda_device_number=0 --exp_id=$3
elif [ $1 = "rollout" ]; then
    echo "Rollout"
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="$2" --output_path=${ROLLOUT_PATH} --mode='rollout' --cuda_device_number=0 --exp_id=$3
elif [ $1 = "render" ]; then
    echo "Rendering"
    python -m gns.render_rollout --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout-$3-$2" --exp_id=$4
elif [ $1 = "all" ]; then
    echo "ALL"
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="latest" --train_state_file="latest" --ntraining_steps=${NTRAINING_STEPS} --nvalid_steps=${NVALID_STEPS} --mode='train' --exp_id=$4
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="$2" --mode="valid" --cuda_device_number=0 --exp_id=$4
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="$2" --output_path=${ROLLOUT_PATH} --mode='rollout' --cuda_device_number=0 --exp_id=$4
    python -m gns.render_rollout --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout-$3-$2" --exp_id=$4
elif [ $1 = "plot" ]; then
    echo "Ploting"
    python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="$2" --output_path=${ROLLOUT_PATH} --mode='rollout' --cuda_device_number=0 --exp_id=$4
    python -m gns.render_rollout --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout-$3-$2" --exp_id=$4
else
    echo "Invalid argument"
fi

