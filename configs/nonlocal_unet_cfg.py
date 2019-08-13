
model=dict(
    model_name='nonlocal_unet',
    params={
        'in_chans': 1,
        'out_chans': 1,
        'chans': 32,
        'num_pool_layers': 4,
        'drop_prob': 0.0
    }
)

data=dict(
    train=dict(
        type='slice',
        data_path='data/singlecoil_train',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=16,
        use_seed=False,
        crop=True, 
        crop_size=160
    ),
    val=dict(
        type='slice',
        data_path='data/singlecoil_val',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=4,
        use_seed=True,
        crop=False, 
        crop_size=128
    )
)

device='cuda'

exp_dir='exp_dir/nonlocal_unet/'
train_cfg=dict(
    data_parallel=False,
    optimizer=dict(
        name='RMSprop',
        params={
            'lr':1e-3,
            'weight_decay':0.
        }
    ),
    lr_scheduler={
        'step_size': 40,
        'gamma': 0.1
    },
    resume=False,
    ckpt=exp_dir+'/model.pt',
    num_epochs=50
)

infer_cfg=dict(
    mask_kspace=True,
    data_path='data/singlecoil_val/',
    center_fractions=[0.08],
    accelerations=[4],
    challenge='singlecoil',
    resolution=320,
    batch_size=4,
    ckpt=exp_dir+'best_model.pt',
    out_dir='/home/ubuntu/BigVolume/tmp/',
    device=device
)