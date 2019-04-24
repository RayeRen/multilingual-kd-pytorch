# Multilingual NMT with Knowledge Distillation on Fairseq

The implementation of [Multilingual Neural Machine Translation with Knowledge Distillation [ICLR2019] ](https://arxiv.org/abs/1902.10461) (Xu Tan*, Yi Ren*, Di He, Tao Qin, Zhou Zhao, Tie-Yan Liu)

> This code is based on [Fairseq](https://github.com/pytorch/fairseq)

## Preparation

1. `pip install -r requirements.txt`
2. `cd data/iwslt/raw; bash prepare-iwslt14.sh`
3. `python setup.py install`

## Run Multilingual NMT with Knowledge Distillation

### Train Experts

1. Run `data_dir=iwslt exp_name=train_expert_LNG1 targets="LNG1" hparams=" --save-output --share-all-embeddings" bash runs/train.sh`.
2. Replace LNG1 with other languages to train all the experts(LNG2, LNG3, ...).
3. Topk output binary files will be produced after steps 1 and 2 in $data/data-bin

### Train Multilingual Student

4. Run `exp_name=train_kd_multilingual targets="LNG1,LNG2 ...(filling with all languages)" hparams=" --share-all-embeddings" bash runs/train_distill.sh` to train the KD multilingual model. BLEU scores will be printed to console every 3 epochs.
