#!/bin/bash

module load python/3.6

export PYTHONPATH=$PYTHONPATH:/home/lambalex/fairseq/

python3 fairseq_cli/train.py  --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir /scratch/lambalex/checkpoints/transformer_wikitext-103/$SLURM_JOB_ID \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000 \
  --no-epoch-checkpoints \
  --seed 1048


