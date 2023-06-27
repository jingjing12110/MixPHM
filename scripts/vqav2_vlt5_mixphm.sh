#!/usr/bin/env bash

# *********************************************************
data_name=vqav2

r=12
e=4
rank=8
n=4
w=0.2

for k in 16 32 64 100 500 1000;do
  name=epoch1000_bs16_lr5e-3_k${k}_r${r}_rank${rank}_n${n}_expert${e}+Lra_w${w}_D1D2SAdn
  output=Exp-VLT5-MixPHM/${data_name}/$name
  mkdir -p $output
  cp $0 $output/run.sh
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    torchrun --nproc_per_node=$2 \
    src/vl_t5/train.py \
    --distributed \
    --data_name $data_name \
    --k $k \
    --seeds "13, 21, 42, 87, 100" \
    --apply_phm_moe True \
    --hypercomplex_adapter True \
    --hypercomplex_division $n \
    --learn_phm True \
    --shared_phm_rule True \
    --factorized_phm True \
    --phm_rank $rank \
    --reduction_factor $r \
    --num_experts $e \
    --inference_level 3 \
    --sharing_down 1 \
    --sharing_up 0 \
    --use_adapter_r2loss True \
    --using_mi_loss True \
    --red_w $w \
    --align_w $w \
    --use_adapter_cross_attn False \
    --use_adapter_self_attn False \
    --lr 5e-3 \
    --batch_size 16 \
    --epochs 1000 \
    --feat_type roi \
    --optim adamw \
    --warmup_ratio 0.05 \
    --clip_grad_norm 5 \
    --num_workers 8 \
    --backbone 't5-base' \
    --output $output \
    --num_beams 5 \
    --valid_batch_size 512 \
    --load snap/pretrain/VLT5-novqa/Epoch30_base
done
