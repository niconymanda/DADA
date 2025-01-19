nohup python main.py --learning_rate 1e-05  --batch_size 32 --margin 0.4   --weight_decay 0.1 --hidden_layers "-1, -2, -3" --mlp_layers 3 > out_0.txt 2>&1 
disown			

nohup python main.py --learning_rate 1e-04  --batch_size 16 --margin 0.5   --weight_decay 0.1 --hidden_layers "-1, -2, -3, -4" --mlp_layers 5 > out_1.txt 2>&1 
disown	

nohup python main.py --learning_rate 1e-04  --batch_size 16 --margin 0.5   --weight_decay 0.1 --hidden_layers "-1, -2" --mlp_layers 4 > out_2.txt 2>&1 
disown	
					