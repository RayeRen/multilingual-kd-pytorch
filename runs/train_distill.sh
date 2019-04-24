#!/usr/bin/env bash

arch=${arch:-transformer_iwslt_de_en_small}
exp_name=${exp_name:-train_expert}
extra_args=${extra_args:-}
data=${data:-iwslt}
sources=${sources:-}
targets=${targets:-en}


export PYTHONPATH=.

python train.py data/${data}/data-bin \
  -a $arch \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --log-format json \
  --save-interval-updates 2000 \
  --log-interval 50 \
  --dropout 0.3 --weight-decay 0.0 --criterion distill_label_smoothed_cross_entropy --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --early-stop 100 \
  --sources=$sources --targets=$targets \
  --max-update 300000 \
  --save-dir checkpoints/$exp_name $extra_args
