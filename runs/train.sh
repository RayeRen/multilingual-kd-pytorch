#!/usr/bin/env bash

data_dir=${data_dir:-iwslt}
exp_name=${exp_name:-test1}
hparams=${hparams:-}
arch=${arch:-transformer_iwslt_de_en_small}
update_freq=${update_freq:-1}
max_tokens=${max_tokens:-4096}
sources=${sources:-}
targets=${targets:-en}

export PYTHONPATH=.

python train.py data/${data_dir}/data-bin \
  -a $arch \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --log-format json \
  --save-interval-updates 2000 \
  --log-interval 50 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --decoder-lng-embed \
  --early-stop 20 \
  --sources=$sources --targets=$targets \
  --max-tokens=$max_tokens --update-freq=$update_freq \
  --max-update 300000 \
  --save-dir checkpoints/$exp_name $hparams
