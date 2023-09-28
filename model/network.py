from torch.optim import lr_scheduler
from .net_architecture import *
from .network_utils import *


######################################################################################
# Define networks
######################################################################################

def define_HeatMap(opt, model):
    if model == 'egoglass':
        net = HeatMap_EgoGlass(opt)
    elif model == "ego3dpose_heatmap_shared":
        net = HeatMap_UnrealEgo_Shared(opt)
    elif model == "ego3dpose_autoencoder":
        net = HeatMap_UnrealEgo_Shared(opt)
    else:
        raise Exception("HeatMap is not implemented for {}".format(model))

    print_network_param(net, 'HeatMap_Estimator for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, opt.init_ImageNet)

def define_AutoEncoder(opt, model):
    if model == 'egoglass':
        net = AutoEncoder(opt, input_channel_scale=2)
    elif model == "ego3dpose_autoencoder":
        net = Ego3DAutoEncoder(opt, input_channel_scale=2)
    else:
        raise Exception("AutoEncoder is not implemented for {}".format(model))

    print_network_param(net, 'AutoEncoder for {}'.format(model))

    return init_net(net, opt.init_type, opt.gpu_ids, False)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters_step, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False
