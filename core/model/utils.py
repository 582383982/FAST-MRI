import torch
import importlib

model_dict = {
    'baseline_unet': 'core.model.unet_model'
}

def build_model(args):
    model_name = args.model.model_name
    params = args.model.params
    if model_name not in model_dict.keys():
        raise 'No such model'
    modlue = importlib.import_module(model_dict[model_name])
    model = modlue.Model(**params).to(args.device)
    return model