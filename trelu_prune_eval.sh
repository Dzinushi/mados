config="marinext.tiny.240x240.mados.py"
model_path="china_results/use_default_decay/trelu_ln/original"
path="./data/MADOS/"
prune_threshold=1e-20
python -m marinext.trelu_prune_eval --path $path --model_path $model_path --config $config --prune_threshold $prune_threshold --split "test" > test.txt
python -m marinext.trelu_prune_eval --path $path --model_path $model_path --config $config --prune_threshold $prune_threshold --split "val" > val.txt
