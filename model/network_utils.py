import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import Adam
from collections import OrderedDict
from dadaptation import DAdaptSGD, DAdaptAdam, DAdaptAdaGrad

######################################################################################
# Functions
######################################################################################
def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_optimizer(opt, module):
    if opt.optimizer == 'adam':
        optimizer = Adam(params=module.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            # eps=1e-4,
        )
    elif opt.optimizer == 'dadam':
        optimizer = DAdaptAdam(params=module.parameters(),
            lr=1.0,
            weight_decay=opt.weight_decay,
            eps=1e-4,
        )
    elif opt.optimizer == 'dsgd':
        optimizer = DAdaptSGD(params=module.parameters(),
            lr=1.0,
            weight_decay=opt.weight_decay,
        )
    elif opt.optimizer == 'dadagrad':
        optimizer = DAdaptAdaGrad(params=module.parameters(),
            lr=1.0,
            weight_decay=opt.weight_decay,
        )
    else:
        print("Invalid Optimizer Option!")
        exit()
    return optimizer
    
def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, gain, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network_param(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('total number of parameters of {}: {:.3f} M'.format(name, num_params / 1e6))


def init_net(net, init_type='normal', gpu_ids=[], init_ImageNet=True):

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()

    if init_ImageNet is False:
        init_weights(net, init_type)
    else:
        init_weights(net.after_backbone, init_type)
        print('   ... also using ImageNet initialization for the backbone')

    return net
        


######################################################################################
# Basic Operation
######################################################################################


def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_deconv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_fc_layer(in_feature, out_feature, with_relu=True, with_bn=True):
    modules = OrderedDict()
    fc = torch.nn.Linear(in_feature, out_feature)
    modules['fc'] = fc
    bn = torch.nn.BatchNorm1d(num_features=out_feature)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)

    if with_bn is True:
        modules['bn'] = bn
    else:
        print('no bn')

    if with_relu is True:
        modules['relu'] = relu
    else:
        print('no pose relu')

    return torch.nn.Sequential(modules)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
