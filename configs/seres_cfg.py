
model=dict(
    model_name='se_res',
    params={
        'in_chans': 1,
        'out_chans': 1,
        'n_resblocks': 8,
        'n_feats': 32
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
        crop=False, 
        crop_size=48
    ),
    val=dict(
        type='slice',
        data_path='data/singlecoil_val',
        challenge='singlecoil',
        resolution=320,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        sample_rate=1.0,
        batch_size=16,
        use_seed=True,
        crop=False, 
        crop_size=48,
        val_count=512
    )
)

device='cuda'

exp_dir='exp_dir/se_res/'
train_cfg=dict(
    data_parallel=True,
    optimizer=dict(
        name='RMSprop',
        params={
            'lr':1e-4,
            'weight_decay':0.
        }
    ),
    lr_scheduler={
        'step_size': 20,
        'gamma': 0.1
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