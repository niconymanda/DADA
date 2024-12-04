nohup python src/authorship_attribution/main.py --clip_grad 50 --epochs 50 --gpu_id 2 > out.txt 2>&1 &
disown

nohup python src/authorship_attribution/main.py --clip_grad 100 --epochs 50 --gpu_id 2 > out_2.txt 2>&1 &
disown

nohup python src/authorship_attribution/main.py --clip_grad 150 --epochs 50 --gpu_id 2 > out_3.txt 2>&1 &
disown