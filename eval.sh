# infer
python infer.py --cfg configs/attunet_cfg.py --mask-kspace --data-parallel
# eval
# python infer.py --cfg configs/mdn_cfg.py --data-parallel
python eval.py --target-path data/singlecoil_val/ \
    --predictions-path /home/ubuntu/BigVolume/tmp/ \
    --challenge singlecoil
    # --acquisition CORPD_FBK
    # --acquisition CORPDFS_FBK