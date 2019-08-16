
model=dict(
    model_name='vnet',
    params={
        'inChans':1,
        'startChans':16,
        'elu': True,
        'nll': False
    }
)

data=dict(
    train=dict(
        type='3d',
        data_path='data/singlecoil_train',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=1,
        use_seed=False,
        crop=True, 
        crop_size=96
    ),
    val=dict(
        type='3d',
        data_path='data/singlecoil_val',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=1,
        use_seed=True,
        crop=False, 
        crop_size=48
    )
)

device='cuda'

exp_dir='exp_dir/vnet/'
train_cfg=dict(
    data_parallel=True,
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
    ckpt=exp_dir +'model.pt',
    num_epochs=50
)

infer_cfg=dict(
    mask_kspace=True,
    data_path='data/singlecoil_val/',
    center_fractions=[0.04],
    accelerations=[8],
    challenge='singlecoil',
    resolution=320,
    batch_size=16,
    ckpt=exp_dir+'best_model.pt',
    out_dir=exp_dir+'infer/',
    device=device
)