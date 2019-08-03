import torch
import importlib
from torch import nn
model_dict = {
    'baseline_unet': {'pkg':'core.model.unet_model', 'name': 'UnetModel'},
    'cprn': {'pkg':'core.model.cprn', 'name': 'cprn'},
    'dbpn': {'pkg':'core.model.dbpn.dbpn_model', 'name': 'dbpn'},
    'se_unet': {'pkg':'core.model.senet.se_unet', 'name': 'seunet'},
    'vnet':{'pkg':'core.model.vnet', 'name': 'vnet'},
    'se_res':{'pkg':'core.model.senet.se_resnet', 'name': 'seres'},
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
    model.apply(init_model)
    return model

def init_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()