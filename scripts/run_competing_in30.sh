CUDA_VISIBLE_DEVICES=$2 python main.py --dataset imagenet --out $1 --arch SCOMatch_resnet_imagenet \
--batch-size 64 --lr 0.03 --expand-labels --seed 1 --opt_level O2 --mu 2 --epochs 512 --threshold 0.95 --ood-threshold 0.95 --start_fix 0 --Km 1
# sh run_competing_in30.sh scomatch_in30 GPU




