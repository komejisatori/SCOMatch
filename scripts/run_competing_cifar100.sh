CUDA_VISIBLE_DEVICES=$5 python main.py --dataset cifar100 --num-labeled $1 --out $2 --num-super $3 --arch SCOMatch_wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed $4 --opt_level O2  --mu 2 --threshold 0.95 --start_fix 0 --ood-threshold 0.95 --Km 1
# sh run_competing_cifar100.sh 25 scomatch_cifar100_25 11 SEED GPU_ID










