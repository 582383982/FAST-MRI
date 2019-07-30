# infer
# python infer.py --cfg configs/baseline_unet.py --mask-kspace --data-parallel
# eval
python eval.py --target-path data/singlecoil_val/ \
    --predictions-path exp_dir/baseline_unet/infer \
    --challenge singlecoil \
    --acquisition CORPDFS_FBK
    # --acquisition CORPDFS_FBK