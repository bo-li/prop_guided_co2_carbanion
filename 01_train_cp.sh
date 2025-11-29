#!/bin/bash
#
# Train Chemprop model with SMILES + solvent features
#
echo "====================================================================================================
GENERATING SOLVENT FEATURES
====================================================================================================
"
python generate_solvent_features.py \
    --data_csv ./data/mayr_nucleophiles.csv \
    --out_csv ./data/solvent_features.csv

echo "====================================================================================================
TRAINING CHEMPROP MODEL: SMILES + SOLVENT FEATURES
====================================================================================================
"
save_dir=./models/cp_model

# Training command 
chemprop_train \
  --data_path ./data/mayr_nucleophiles.csv \
  --dataset_type regression \
  --target_columns N sN \
  --save_dir "$save_dir" \
  --smiles_columns smiles \
  --epochs 100 \
  --hidden_size 300 \
  --ffn_hidden_size 300 \
  --depth 3 \
  --save_smiles_splits \
  --seed 42 \
  --features_path ./data/solvent_features.csv \
  2>&1 | tee train_cp.log

echo "
====================================================================================================
TRAINING COMPLETE
====================================================================================================
"

