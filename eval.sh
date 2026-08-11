config="marinext.tiny.240x240.mados_aln_abn.py"
model_path="china_results/no_decay/trelu_aln_abn/original"
path="./data/MADOS/"
#python -m marinext.evaluation --path $path --model_path $model_path --config $config --split "test" > trelu.txt
python -m marinext.evaluation --path $path --model_path $model_path --config $config --split "test" > test.txt
python -m marinext.evaluation --path $path --model_path $model_path --config $config --split "val" > val.txt
