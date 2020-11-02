#!/bin/bash

module load python/3.6

export PYTHONPATH=$PYTHONPATH:/home/lambalex/fairseq/

#512
#1024
#512
#1024

#712
#1424

rm -rf checkpoints/

#embdim=712
#ffndim=1424

embdim=512
ffndim=1024

#CUDA_VISIBLE_DEVICES=0
python3 fairseq_cli/train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir /scratch/lambalex/checkpoints/transformer_wikitext-103/$SLURM_JOB_ID \
    --encoder-embed-dim	$embdim \
    --encoder-ffn-embed-dim $ffndim \
    --decoder-embed-dim	$embdim \
    --decoder-ffn-embed-dim $ffndim \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --topk_ratio 1.0 \
    --num_modules 4 \
    --use_module_communication true



#> out.txt
