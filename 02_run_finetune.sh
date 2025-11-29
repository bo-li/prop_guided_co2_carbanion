#!/bin/bash
# Finetune generator with hybrid approach: log_k predictor + regex carbanion filter

python finetune_generator_carbanion_hybrid.py \
  --train ./data/seed_carbanions.smi \
  --vocab ./data/chembl_vocab.txt \
  --generative_model ./hgraph2graph/ckpt/chembl-pretrained/model.ckpt \
  --chemprop_model ./models/cp_model_ref \
  --save_dir ./models/finetune_model \
  --solvent DMSO \
  --logk_min -6.0 \
  --logk_max 8.0 \
  --min_similarity 0.1 \
  --max_similarity 0.5 \
  --nsample 100 \
  --epoch 10 \
  --batch_size 50 \
  --lr 1e-3 \
  --seed 42 \
  2>&1 | tee finetune_out.log
