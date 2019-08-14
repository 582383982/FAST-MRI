# infer
# python infer.py --cfg configs/dilated_cat_unet.py --mask-kspace --data-parallel
# eval
# python infer.py --cfg configs/mdn_cfg.py --data-parallel
python eval.py --target-path data/singlecoil_val/ --predictions-path data/tmp/ --challenge singlecoil
python eval.py --target-path data/singlecoil_val/ --predictions-path data/tmp/ --challenge singlecoil --acquisition CORPD_FBK
python eval.py --target-path data/singlecoil_val/ --predictions-path data/tmp/ --challenge singlecoil --acquisition CORPDFS_FBK