import torch
import importlib

model_dict = {
    'baseline_unet': {'pkg':'core.model.unet_model', 'name': 'UnetModel'},
    'cprn': {'pkg':'core.model.cprn', 'name': 'cprn'},
    'dbpn': {'pkg':'core.model.dbpn.dbpn_model', 'name': 'dbpn'}
}

def build_model(args):
    model_name = args.model.model_name
    params = args.model.params
    if model_name not in model_dict.keys():
        raise 'No such model'
    module = importlib.import_module(model_dict[model_name]['pkg'])
    Model = module.get_model(model_dict[model_name]['name'])
    model = Model(**params)
    model.to(args.device)
    return model