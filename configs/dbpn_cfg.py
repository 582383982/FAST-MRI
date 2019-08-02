
model=dict(
    model_name='dbpn',
    params={
        'num_channels': 1, 
        'base_filter': 32, 
        'feat': 64, 
        'num_stages': 6, 
        'scale_factor': 2
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
        crop_size=80
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
        crop_size=80,
        val_count=512
    )
)

device='cuda'

exp_dir='exp_dir/dbpn/'
train_cfg=dict(
    data_parallel=True,
    optimizer=dict(
        name='RMSprop',
        params={
            'lr':1e-5,
            'weight_decay':0.
        }
    ),
    lr_scheduler={
        'step_size': 20,
        'gamma': 0.2
    },
    resume=False,
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