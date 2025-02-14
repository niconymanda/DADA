
nohup python ./authorship_attribution/main.py --model_name 'microsoft/deberta-v3-large' --learning_rate 1e-05 --margin 0.5  --batch_size 16  --hidden_layers "-1" --mlp_layers 1 --data "/data/iivanova-23/data/wiki/wiki_scrape_truncate.csv" "/data/amathur-23/DADA/VoxCeleb2/" --gpu_id '2' > deberta_scrape.txt 2>&1 &
disown	

# nohup python ./authorship_attribution/main.py --model_name 'microsoft/deberta-v3-large' --learning_rate 1e-05 --margin 0.5  --batch_size 16  --hidden_layers "-1" --mlp_layers 1 --data "/data/amathur-23/DADA/VoxCeleb2/" > deberta_2.txt 2>&1 
# disown	

# nohup python ./authorship_attribution/main.py --data "/data/iivanova-23/data/wild_wiki.csv" "/data/amathur-23/DADA/VoxCeleb2/" --model_name 'google-t5/t5-large' --learning_rate 1e-04 --margin 0.44 --gpu_id '2' --hidden_layers "-1" --mlp_layers 2 > t5_2.txt 2>&1 &
# disown

# nohup python ./authorship_attribution/main.py --model_name 'microsoft/deberta-v3-large' --learning_rate 1e-05 --margin 0.5  --batch_size 16  --hidden_layers "-1" --mlp_layers 1 --data "/data/iivanova-23/data/wild_wiki.csv" "/data/iivanova-23/data/VoxCeleb2/" --gpu_id '2' > deberta_3.txt 2>&1 &
# disown





