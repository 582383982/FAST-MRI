# infer
# python infer.py --cfg configs/baseline_unet.py --mask-kspace --data-parallel
# eval
# python infer.py --cfg configs/mdn_cfg.py --data-parallel
python eval.py --target-path data/singlecoil_val/ \
    --predictions-path exp_dir/mdn_net/infer \
    --challenge singlecoil
    # --acquisition CORPD_FBK
    # --acquisition CORPDFS_FBK