#!/bin/bash
python -m marinext.train --epochs 120 --reduce_lr_on_plateau 1 --factor 0.7 --patience 5 --batch 8 --num_workers 6 --decay 0.0 --path ./data/MADOS/ --config marinext.tiny.240x240.mados_aln_abn_agn.py
