config="marinext.tiny.240x240.mados.py"
model_path="china_results/use_default_decay/trelu_ln/original_pruned"
prune_threshold=1e-20
apply_pruning="false"
python -m marinext.model_summary \
  --model_path=$model_path \
  --config=$config \
  --apply_pruning=$apply_pruning \
  --prune_threshold=$prune_threshold > model_summary_pruned.txt
