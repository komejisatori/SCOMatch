CUDA_VISIBLE_DEVICES=$5 python main.py --dataset tinyimagenet --num-labeled $1 --out $2 --num-super $3 --arch SCOMatch_wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed $4 --opt_level O2 --mu 3 --threshold 0.95 --start_fix 0 --ood-threshold 0.95
# sh run_competing_tiny.sh 100 scomatch_tiny_100 100 SEED GPU_ID