from torch import optim
def build_optimizer(cfg, model_params):
    optimizer = None
    if cfg.name == 'Adam':
        optimizer = optim.Adam
    else:
        optimizer = optim.RMSprop
    return optimizer(model_params, **cfg.params)

