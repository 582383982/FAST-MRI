
model=dict(
    model_name='cprn',
    params={
        'in_chans': 1,
        'out_chans': 1,
        'num_cp': 2, 
        'cp_chans': 32, 
        'num_res': 8, 
        'res_chans': 64
    }
)

data=dict(
    train=dict(
        data_path='data/singlecoil_train',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=16,
        use_seed=False,
        crop=True, 
        crop_size=48
    ),
    val=dict(
        data_path='data/singlecoil_val',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=16,
        use_seed=True,
        crop=False, 
        crop_size=48
    )
)

device='cuda'

exp_dir='exp_dir/cprn/'
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
    resume=True,
    ckpt=exp_dir+'model.pt',
    num_epochs=50
)

infer_cfg=dict(
    mask_kspace=True,
    data_path='data/singlecoil_val/',
    center_fractions=[0.08],
    accelerations=[4],
    challenge='singlecoil',
    resolution=320,
    batch_size=16,
    ckpt=exp_dir+'best_model.pt',
    out_dir=exp_dir+'infer/',
    device=device
)