
nohup python main.py --model_name 'answerdotai/ModernBERT-large' --learning_rate 1e-04 --margin 0.5  --batch_size 16  --hidden_layers "-1, -2, -3" --mlp_layers 3 > ./output/out_txt/bert_1.txt 2>&1 
disown	

nohup python main.py --model_name 'answerdotai/ModernBERT-large' --learning_rate 1e-04 --margin 0.4  --batch_size 16  --hidden_layers "-1" --mlp_layers 3 > ./output/out_txt/bert_2.txt 2>&1 
disown	

nohup python main.py --model_name 'google/flan-t5-large' --learning_rate 1e-04 --margin 0.35  --batch_size 16  --hidden_layers "-1" --mlp_layers 5 > ./output/out_txt/flan_1.txt 2>&1 
disown	

nohup python main.py --model_name 'microsoft/deberta-v3-large' --learning_rate 1e-05 --margin 0.5  --batch_size 16  --hidden_layers "-1" --mlp_layers 1 > ./output/out_txt/deberta_1.txt 2>&1 
disown