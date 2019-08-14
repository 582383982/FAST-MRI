sleep 1m
python train.py --cfg configs/dilated_add_unet.py
sleep 15m
python train.py --cfg configs/dilated_cat_unet.py
sleep 15m
python train.py --cfg configs/baseline_unet_adam.py 