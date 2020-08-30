#!/bin/bash

module load python/3.6

export PYTHONPATH=$PYTHONPATH:/home/lambalex/fairseq/

#512
#1024
#512
#1024

#712
#1424

rm -rf checkpoints_2mod/

#embdim=712
#ffndim=1424

embdim=512
ffndim=1024

#CUDA_VISIBLE_DEVICES=0
fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints_2mod/ \
    --max-tokens 4000 \
    --update-freq 2 \
    --no-epoch-checkpoints \
    --topk_ratio 1.0 \
    --num_modules 2 \
    --encoder-embed-dim $embdim \
    --encoder-ffn-embed-dim $ffndim \
    --decoder-embed-dim $embdim \
    --decoder-ffn-embed-dim $ffndim \
    --use_module_communication true \
    --use_value_competition true

#8 / num_gpus

#> out.txt
