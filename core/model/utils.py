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
    'mdn_net':{'pkg':'core.model.mdn_net', 'name': 'mdn_net'},
    'antialiase_unet':{'pkg':'core.model.antialiase.model', 'name': 'UnetModel'},
    'nonlocal_unet':{'pkg':'core.model.non_local.model', 'name': 'UnetModel'},
    'attention_unet':{'pkg':'core.model.attention_unet', 'name': 'UnetModel'},
    'dilated_unet_add':{'pkg':'core.model.receptive_field.dilated_unet_add', 'name': 'UnetModel'},
    'dilated_unet_cat':{'pkg':'core.model.receptive_field.dilated_unet_cat', 'name': 'UnetModel'},
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