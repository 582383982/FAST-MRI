
model=dict(
    model_name='baseline_unet',
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
        data_path='data/singlecoil_train',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=16,
        use_seed=False,
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
    )
)

device='cuda'

exp_dir='exp_dir/baseline_unet/'
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
    ckpt='exp_dir/baseline_unet/model.pt',
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