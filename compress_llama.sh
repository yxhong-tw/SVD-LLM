#!/bin/bash

set -e

FINE_TUNE_PATH="."
MODEL_PATH="/workspace/models/Llama-2-7b-hf"
DEV="cuda:0"

python3 SVDLLM.py \
    --model $MODEL_PATH \
    --step 1 \
    --ratio 0.5 \
    --whitening_nsamples 256 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path . \
    --DEV $DEV
python3 SVDLLM.py \
    --step 4 \
    --model_path _workspace_models_Llama_2_7b_hf_whitening_only_0.5.pt \
    --DEV $DEV

python3 utils/LoRA.py \
    --prune_model _workspace_models_Llama_2_7b_hf_whitening_only_0.5.pt \
    --data_path yahma/alpaca-cleaned \
    --output_dir $FINE_TUNE_PATH/first_half \
    --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj \
    --lora_r 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64
python3 SVDLLM.py \
    --model_path _workspace_models_Llama_2_7b_hf_whitening_only_0.5.pt \
    --lora $FINE_TUNE_PATH/first_half /first_half \
    --step 4 \
    --DEV $DEV

python3 utils/LoRA.py \
    --prune_model $FINE_TUNE_PATH/first_half/merge.pt \
    --data_path yahma/alpaca-cleaned \
    --output_dir $FINE_TUNE_PATH/second_half \
    --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj \
    --lora_r 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 64
python3 SVDLLM.py \
    --model_path _workspace_models_Llama_2_7b_hf_whitening_only_0.5.pt \
    --lora $FINE_TUNE_PATH/first_half /first_half \
    --step 4 \
    --DEV $DEV

python3 SVDLLM.py \
    --model_path $FINE_TUNE_PATH/first_half/merge.pt \
    --lora $FINE_TUNE_PATH/second_half \
    --step 4 \
    --DEV $DEV
