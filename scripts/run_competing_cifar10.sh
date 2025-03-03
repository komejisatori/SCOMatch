CUDA_VISIBLE_DEVICES=$4 python main.py --dataset cifar10 --num-labeled $1 --out $2 --arch SCOMatch_wideresnet \
--batch-size 64  --expand-labels --seed $3 --opt_level O2  --mu 2 --threshold 0.95 --start_fix 0 --ood-threshold 0.95 --Km 1
# sh run_competing_cifar10.sh 25 scomatch_cifar10_25 SEED GPU_ID





