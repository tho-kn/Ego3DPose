import torch
import torch.nn as nn
from torchvision import models
import math
from .network_utils import *
from utils.util import get_kinematic_parents

######################################################################################
# Network structure
######################################################################################

def get_limb_dim(opt):
    if opt.heatmap_type == 'none':
        limb_heatmap_dim = 0
    elif opt.heatmap_type == 'sin':
        limb_heatmap_dim = 2
    else: # 'angle', 'depth'
        limb_heatmap_dim = 1
        
    if opt.use_limb_heatmap:
        limb_heatmap_dim += 1
        
    return limb_heatmap_dim
            
############################## EgoGlass ##############################

class HeatMap_EgoGlass(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_EgoGlass, self).__init__()

        self.backbone = HeatMap_EgoGlass_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_EgoGlass_AfterBackbone(opt)

    def forward(self, input):
        
        x = self.backbone(input)
        output, segmentation = self.after_backbone(x)

        return output, segmentation


class HeatMap_EgoGlass_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_EgoGlass_Backbone, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output


class HeatMap_EgoGlass_AfterBackbone(nn.Module):
    def __init__(self, opt):
        super(HeatMap_EgoGlass_AfterBackbone, self).__init__()

        self.num_heatmap = opt.num_heatmap
        self.num_seg = 4

        self.input_1x1 = convrelu(3, 64, 1, 0)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up_input = convrelu(64 + 256, 256, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_heatmap = nn.Conv2d(256, self.num_heatmap, 1)
        self.conv_seg = nn.Conv2d(256, self.num_seg, 1)


    def forward(self, list_input):
        input = list_input[0]
        layer0 = list_input[1]
        layer1 = list_input[2]
        layer2 = list_input[3]
        layer3 = list_input[4]
        layer4 = list_input[5]
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        output = self.conv_heatmap(x)
        
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        input = self.input_1x1(input)
        x = torch.cat([x, input], dim=1)
        x = self.conv_up_input(x)
        
        segmentation = self.conv_seg(x)

        return output, segmentation


############################## UnrealEgo ##############################

class HeatMap_UnrealEgo_Shared(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared, self).__init__()

        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name)

    def forward(self, input_left, input_right, prev_pose=None):

        x_left, x_right = self.backbone(input_left, input_right)
        output = self.after_backbone(x_left, x_right, pose=prev_pose)

        return output

class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()
        self.joint_preset = opt.joint_preset
        self.use_separate_encoder = False
        
        if self.use_separate_encoder:
            self.backbone_l = Encoder_Block(opt, model_name=model_name)
            self.backbone_r = Encoder_Block(opt, model_name=model_name)
            return
            
        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, input_left, input_right):
        if self.use_separate_encoder:
            output_left = self.backbone_l(input_left)
            output_right = self.backbone_r(input_right)
            return output_left, output_right
        
        output_left = self.backbone(input_left)
        output_right = self.backbone(input_right)
        return output_left, output_right

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = convrelu(in_channels, out_channels, kernel_size, padding)
        self.conv2 = convrelu(out_channels, out_channels, kernel_size, padding)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    def __init__(self, opt, model_name="resnet18"):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        if model_name == 'resnet18':
            feature_scale = 1
        elif model_name == "resnet34":
            feature_scale = 1
        elif model_name == "resnet50":
            feature_scale = 4
        elif model_name == "resnet101":
            feature_scale = 4
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        limb_heatmap_dim = get_limb_dim(opt)
        self.num_heatmap = opt.num_heatmap + opt.num_rot_heatmap * limb_heatmap_dim
        
        # self.layer0_1x1 = convrelu(128, 128, 1, 0)
        self.layer1_1x1 = convrelu(128 * feature_scale, 128 * feature_scale, 1, 0)
        self.layer2_1x1 = convrelu(256 * feature_scale, 256 * feature_scale, 1, 0)
        self.layer3_1x1 = convrelu(512 * feature_scale, 516 * feature_scale, 1, 0)
        self.layer4_1x1 = convrelu(1024 * feature_scale, 1024 * feature_scale, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        conv_up3_in_ch = 516 * feature_scale + 1024 * feature_scale
        conv_up2_in_ch = 256 * feature_scale + 1024 * feature_scale
        conv_up1_in_ch = 128 * feature_scale + 512 * feature_scale
            
        self.conv_up3 = convrelu(conv_up3_in_ch, 1024 * feature_scale, 3, 1)
        self.conv_up2 = convrelu(conv_up2_in_ch, 512 * feature_scale, 3, 1)
        self.conv_up1 = convrelu(conv_up1_in_ch, 512 * feature_scale, 3, 1)

        self.conv_heatmap = nn.Conv2d(512 * feature_scale, self.num_heatmap * 2, 1)

    def forward(self, list_input_left, list_input_right, pose=None):
        list_stereo_feature = [
            torch.cat([list_input_left[id], list_input_right[id]], dim=1) for id in range(len(list_input_left))
        ]
                
        input = list_stereo_feature[0] # size = [16, 6, 256, 256]
        layer0 = list_stereo_feature[1] # size = [16, 128, 128, 128]
        layer1 = list_stereo_feature[2] # size = [16, 128, 64, 64]
        layer2 = list_stereo_feature[3] # size = [16, 256, 32, 32]
        layer3 = list_stereo_feature[4] # size = [16, 512, 16, 16]
        layer4 = list_stereo_feature[5] # size = [16, 1024, 8, 8]

        layer4 = self.layer4_1x1(layer4) # size = [16, 1024, 8, 8]
        x = self.upsample(layer4) # size = [16, 1024, 16, 16]
        layer3 = self.layer3_1x1(layer3) # size = [16, 516, 16, 16]
        
        x = torch.cat([x, layer3], dim=1) # size = [16, 1540, 16, 16]
        x = self.conv_up3(x) # size = [16, 1024, 16, 16]

        x = self.upsample(x) # size = [16, 1024, 32, 32]
        layer2 = self.layer2_1x1(layer2) # size = [16, 256, 32, 32]
        
        x = torch.cat([x, layer2], dim=1) # size = [16, 1280, 32, 32]
        x = self.conv_up2(x) # size = [16, 512, 32, 32]

        x = self.upsample(x) # size = [16, 512, 64, 64]
        layer1 = self.layer1_1x1(layer1) # size = [16, 128, 64, 64]
        
        x = torch.cat([x, layer1], dim=1) # size = [16, 640, 64, 64]
        x = self.conv_up1(x) # size = [16, 512, 64, 64]

        output = self.conv_heatmap(x) # size = [16, 30, 64, 64]

        return output


############################## AutoEncoder ##############################


class MLPDecoder(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        ## pose decoder
        super(MLPDecoder, self).__init__()
        
        self.with_bn = True
        self.with_pose_relu = True
        self.pose_fc1 = make_fc_layer(input_dim, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc2 = make_fc_layer(32, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc3 = torch.nn.Linear(32, output_dim)
        
    def forward(self, input):
        input = self.pose_fc1(input)
        input = self.pose_fc2(input)
        return self.pose_fc3(input)
        

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]

        return output


class Ego3DAutoEncoder(nn.Module):
    def __init__(self, opt, input_channel_scale=1, fc_dim=16384):
        super(Ego3DAutoEncoder, self).__init__()

        self.joint_preset = opt.joint_preset
        self.hidden_size = opt.ae_hidden_size
        self.with_bn = True
        self.with_pose_relu = True
        
        self.num_joints = opt.num_heatmap + 1
        if opt.joint_preset == "EgoCap":
            self.num_joints -= 1
        
        self.limb_heatmap_dim = get_limb_dim(opt)

        self.num_rot_heatmap = opt.num_rot_heatmap * self.limb_heatmap_dim
        self.num_heatmap = opt.num_heatmap + self.num_rot_heatmap

        self.input_channel_scale = input_channel_scale
        self.channels_heatmap = self.num_heatmap * input_channel_scale
        self.fc_dim = fc_dim
        self.pose_dim = self.num_joints * 3
        
        self.joint_preset = opt.joint_preset
            
        self.rot_dim = opt.num_rot_heatmap * 3
        if self.limb_heatmap_dim == 0:
            self.rot_dim = opt.num_heatmap * 3 - 3

        conv_ch = [self.channels_heatmap, 64, 128, 256]
        conv_input_ch = conv_ch[:-1].copy()
        conv_output_ch = conv_ch[1:].copy()
        
        self.total_fc_dim = self.fc_dim
        
        rot_conv_ch = conv_ch.copy()
        rot_hidden_size = 10
        rot_conv_input_ch = rot_conv_ch[:-1].copy()
        rot_conv_output_ch = rot_conv_ch[1:].copy()
        rot_decoder_hm_ch = input_channel_scale * self.limb_heatmap_dim
        
        if self.limb_heatmap_dim == 0:
            # Use two positional heatmaps instead of limb heatmaps
            rot_decoder_hm_ch = input_channel_scale * 2
        self.rot_conv1 = make_conv_layer(in_channels=rot_decoder_hm_ch, out_channels=rot_conv_output_ch[0], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.rot_conv2 = make_conv_layer(in_channels=rot_conv_input_ch[1], out_channels=rot_conv_output_ch[1], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.rot_conv3 = make_conv_layer(in_channels=rot_conv_input_ch[2], out_channels=rot_conv_output_ch[2], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        
        rot_fc_dim = rot_conv_output_ch[-1] * 8 * 8
        rot_fc_size = [2048, 512, rot_hidden_size]
        self.rot_fc1 = make_fc_layer(in_feature=rot_fc_dim, out_feature=rot_fc_size[0], with_bn=self.with_bn)
        self.rot_fc2 = make_fc_layer(in_feature=rot_fc_size[0], out_feature=rot_fc_size[0]//4, with_bn=self.with_bn)
        self.rot_fc3 = make_fc_layer(in_feature=rot_fc_size[0]//4, out_feature=rot_hidden_size, with_bn=self.with_bn)
            
        self.conv1 = make_conv_layer(in_channels=conv_input_ch[0], out_channels=conv_output_ch[0], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2 = make_conv_layer(in_channels=conv_input_ch[1], out_channels=conv_output_ch[1], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3 = make_conv_layer(in_channels=conv_input_ch[2], out_channels=conv_output_ch[2], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        self.fc1 = make_fc_layer(in_feature=self.total_fc_dim, out_feature=2048, with_bn=self.with_bn)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=self.with_bn)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=self.hidden_size, with_bn=self.with_bn)

        pose_input_dim = self.hidden_size
        rot_input_dim = self.hidden_size
        
        rot_mlp_output_dim = self.rot_dim
        rot_input_dim = rot_hidden_size
        rot_mlp_output_dim = 3
        pose_input_dim += self.rot_dim
        
        self.rot_mlp = MLPDecoder(opt, rot_input_dim, rot_mlp_output_dim)
        self.pose_mlp = MLPDecoder(opt, pose_input_dim, self.pose_dim)

        # heatmap decoder
        self.heatmap_fc1 = make_fc_layer(self.hidden_size, 512, with_bn=self.with_bn)
        self.heatmap_fc2 = make_fc_layer(512, 2048, with_bn=self.with_bn)
        self.heatmap_fc3 = make_fc_layer(2048, self.fc_dim, with_bn=self.with_bn)
        self.WH = int(math.sqrt(self.fc_dim/256))  

        self.deconv1 = make_deconv_layer(256, 128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv2 = make_deconv_layer(128, 64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv3 = make_deconv_layer(64, self.channels_heatmap, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

    def predict_pose(self, input, input_rgb_left=None, input_rgb_right=None):
        return self.forward(input, input_rgb_left, input_rgb_right, pose_only=True)

    def forward(self, input, input_rgb_left=None, input_rgb_right=None, pose_only=False):
        batch_size = input.size()[0]
        
        # encode heatmap
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)

        z_bar = z
        
        rot_z = z
        
        if self.limb_heatmap_dim == 0:
            hm_left, hm_right = torch.chunk(input, 2, dim=1)
            # Positional heatmap does not include head, index is shifted
            parent_hm_left = torch.stack([hm_left[:, get_kinematic_parents(self.joint_preset)[i]-1, :, :] for i in range(2, hm_left.shape[1]+1)], dim=1)
            parent_hm_right = torch.stack([hm_right[:, get_kinematic_parents(self.joint_preset)[i]-1, :, :] for i in range(2, hm_right.shape[1]+1)], dim=1)
            hm_left, hm_right = hm_left[:, 1:, :, :], hm_right[:, 1:, :, :]
            limb_hm = torch.stack([hm_left, hm_right, parent_hm_left, parent_hm_right], dim=2)
            limb_hm = limb_hm.reshape(-1, limb_hm.shape[2], limb_hm.shape[3], limb_hm.shape[4])
        else:
            limb_heatmaps = input[:, -self.num_rot_heatmap * self.input_channel_scale:]
            limb_hm_left, limb_hm_right = torch.chunk(limb_heatmaps, 2, dim=1)
            
            # batch, hm_dim, joints, h, w
            limb_hm_left = limb_hm_left.reshape(limb_hm_left.shape[0], self.limb_heatmap_dim, -1, limb_hm_left.shape[2], limb_hm_left.shape[3])
            limb_hm_right = limb_hm_right.reshape(limb_hm_right.shape[0], self.limb_heatmap_dim, -1, limb_hm_right.shape[2], limb_hm_right.shape[3])
            limb_hm = torch.cat([limb_hm_left, limb_hm_right], dim=1)
            limb_hm = torch.swapaxes(limb_hm, 1, 2)
            limb_hm = limb_hm.reshape(-1, limb_hm.shape[2], limb_hm.shape[3], limb_hm.shape[4])
        
        rot = self.rot_conv1(limb_hm)
        rot = self.rot_conv2(rot)
        rot = self.rot_conv3(rot)
        rot = rot.view(rot.shape[0], -1)
        rot = self.rot_fc1(rot)
        rot = self.rot_fc2(rot)
        rot_z = self.rot_fc3(rot)

        # decode pose
        rot = self.rot_mlp(rot_z)
        rot = rot.reshape(batch_size, -1)
        z_bar = torch.cat((z_bar, rot.detach().clone()), dim=-1)

        output_pose = self.pose_mlp(z_bar)

        output_pose = output_pose[:, :].view(batch_size, self.num_joints, 3)
        
        if pose_only:
            return output_pose

        # decode heatmap
        x_hm = self.heatmap_fc1(z)
        x_hm = self.heatmap_fc2(x_hm)
        x_hm = self.heatmap_fc3(x_hm) 
        x_hm = x_hm.view(batch_size, 256, self.WH, self.WH)
        x_hm = self.deconv1(x_hm)
        x_hm = self.deconv2(x_hm)
        output_hm = self.deconv3(x_hm)

        return output_pose, rot, output_hm
    
    
class AutoEncoder(nn.Module):
    def __init__(self, opt, input_channel_scale=1, fc_dim=16384, num_heatmap=None, out_dim=None):
        super(AutoEncoder, self).__init__()

        self.hidden_size = opt.ae_hidden_size
        self.with_bn = True
        self.with_pose_relu = True
        
        self.num_joints = opt.num_heatmap + 1
        
        limb_heatmap_dim = get_limb_dim(opt)
        self.num_heatmap = opt.num_heatmap + opt.num_rot_heatmap * limb_heatmap_dim
        if num_heatmap is not None:
            self.num_heatmap = num_heatmap

        self.channels_heatmap = self.num_heatmap * input_channel_scale
        self.fc_dim = fc_dim

        conv_ch = [self.channels_heatmap, 64, 128, 256]
        conv_input_ch = conv_ch[:-1].copy()
        conv_output_ch = conv_ch[1:].copy()
        
        self.out_dim = self.num_joints * 3
        self.output_3d = out_dim is None
        if out_dim is not None:
            self.out_dim = out_dim
        
        self.total_fc_dim = self.fc_dim
            
        self.conv1 = make_conv_layer(in_channels=conv_input_ch[0], out_channels=conv_output_ch[0], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2 = make_conv_layer(in_channels=conv_input_ch[1], out_channels=conv_output_ch[1], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3 = make_conv_layer(in_channels=conv_input_ch[2], out_channels=conv_output_ch[2], kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        self.fc1 = make_fc_layer(in_feature=self.total_fc_dim, out_feature=2048, with_bn=self.with_bn)
        self.fc2 = make_fc_layer(in_feature=2048, out_feature=512, with_bn=self.with_bn)
        self.fc3 = make_fc_layer(in_feature=512, out_feature=self.hidden_size, with_bn=self.with_bn)

        ## pose decoder
        pose_input_dim = self.hidden_size
        self.pose_fc1 = make_fc_layer(pose_input_dim, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc2 = make_fc_layer(32, 32, with_relu=self.with_pose_relu, with_bn=self.with_bn)
        self.pose_fc3 = torch.nn.Linear(32, self.out_dim)

        # heatmap decoder
        self.heatmap_fc1 = make_fc_layer(self.hidden_size, 512, with_bn=self.with_bn)
        self.heatmap_fc2 = make_fc_layer(512, 2048, with_bn=self.with_bn)
        self.heatmap_fc3 = make_fc_layer(2048, self.fc_dim, with_bn=self.with_bn)
        self.WH = int(math.sqrt(self.fc_dim/256))  

        self.deconv1 = make_deconv_layer(256, 128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv2 = make_deconv_layer(128, 64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.deconv3 = make_deconv_layer(64, self.channels_heatmap, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

    def predict_pose(self, input):
        return self.forward(input, pose_only=True)

    def forward(self, input, pose_only=False):
        batch_size = input.size()[0]

        # encode heatmap
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        z = self.fc3(x)

        z_bar = z
        
        # decode pose
        x_pose = self.pose_fc1(z_bar)
        x_pose = self.pose_fc2(x_pose)
        output_pose = self.pose_fc3(x_pose)

        if self.output_3d:
            output_pose = output_pose.view(batch_size, self.num_joints, 3)
        if pose_only:
            return output_pose

        # decode heatmap
        x_hm = self.heatmap_fc1(z)
        x_hm = self.heatmap_fc2(x_hm)
        x_hm = self.heatmap_fc3(x_hm) 
        x_hm = x_hm.view(batch_size, 256, self.WH, self.WH)
        x_hm = self.deconv1(x_hm)
        x_hm = self.deconv2(x_hm)
        output_hm = self.deconv3(x_hm)

        return output_pose, output_hm


if __name__ == "__main__":
    
    model = HeatMap_UnrealEgo_Shared(opt=None, model_name='resnet50')

    input = torch.rand(3, 3, 256, 256)
    outputs = model(input, input)
    pred_heatmap_left, pred_heatmap_right = torch.chunk(outputs, 2, dim=1)

    print(pred_heatmap_left.size())
    print(pred_heatmap_right.size())
