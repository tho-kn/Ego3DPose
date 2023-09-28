import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss

from dadaptation import DAdaptSGD
from .base_model import BaseModel
from . import network
from utils.loss import LossFuncCosSim, LossFuncMPJPE
from utils.util import batch_compute_similarity_transform_torch
import os
import copy

class Ego3DPoseAutoEncoderModel(BaseModel):
    def name(self):
        return 'Ego3DPose AutoEncoder model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = [
            'pose', 'cos_sim', 
        ]

        if self.isTrain:
            self.visual_names = [
                'input_rgb_left', 'input_rgb_right',
            ]
        else:
            self.visual_names = []
            
        if self.opt.num_heatmap > 0:
            self.loss_names.extend([
                'heatmap_left_rec', 'heatmap_right_rec',
            ])
            if self.isTrain:
                self.visual_names.extend([
                    'gt_heatmap_left', 'gt_heatmap_right',
                    'pred_heatmap_left_rec', 'pred_heatmap_right_rec'
                ])
            self.visual_names.extend([
                'pred_heatmap_left', 'pred_heatmap_right',
            ])
        
        if self.opt.num_rot_heatmap > 0:
            self.loss_names.extend([
                'heatmap_limb_left_rec', 'heatmap_limb_right_rec',
            ])
            if self.isTrain:
                self.visual_names.extend([
                    'gt_limb_heatmap_left', 'gt_limb_heatmap_right',
                    'pred_limb_heatmap_left_rec', 'pred_limb_heatmap_right_rec',
                ])
            self.visual_names.extend([
                'pred_limb_heatmap_left', 'pred_limb_heatmap_right',
            ])
            
        self.loss_names.append('rot')

        self.visual_pose_names = [
            "pred_pose", "gt_pose"
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap', 'RotHeatMap', 'AutoEncoder']
        else:
            self.model_names = ['HeatMap', 'RotHeatMap', 'AutoEncoder']
            
        self.opt.indep_rot_decoder = True

        self.eval_key = "mpjpe"
        self.cm2mm = 10
        
        # define the transform network
        pos_opt = copy.deepcopy(opt)
        pos_opt.num_rot_heatmap = 0
        rot_opt = copy.deepcopy(opt)
        rot_opt.num_heatmap = 0
        
        self.net_HeatMap = network.define_HeatMap(pos_opt, model=opt.model)
        self.net_RotHeatMap = network.define_HeatMap(rot_opt, model=opt.model)
        self.net_AutoEncoder = network.define_AutoEncoder(opt, model=opt.model)

        # define loss functions
        self.lossfunc_MSE = MSELoss()
        self.lossfunc_cos_sim = LossFuncCosSim(joint_preset=opt.joint_preset)
        self.lossfunc_MPJPE = LossFuncMPJPE()

        if self.isTrain:
            pretrained_path = opt.path_to_trained_heatmap
            pretrained_dir = os.path.dirname(pretrained_path)
            pretrained_file = os.path.basename(pretrained_path)
            self.load_networks(
                net=self.net_HeatMap, 
                path_to_trained_weights=os.path.join(pretrained_dir + "_pos", pretrained_file)
            )
            net_type = self.opt.heatmap_type
            if net_type == 'none' and self.opt.use_limb_heatmap:
                net_type = 'limb'
            self.load_networks(
                net=self.net_RotHeatMap, 
                path_to_trained_weights=os.path.join(pretrained_dir + "_" + net_type, pretrained_file)
                )
            network._freeze(self.net_HeatMap)
            network._freeze(self.net_RotHeatMap)
            
            # initialize optimizers
            self.optimizers = []
            self.schedulers = []
            
            self.optimizer_AutoEncoder = DAdaptSGD(params=self.net_AutoEncoder.parameters(),
                lr=1.0,
                weight_decay=opt.weight_decay,
                growth_rate=1.02
            )
            self.optimizers.append(self.optimizer_AutoEncoder)

            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)
        self.gt_pose = data['gt_local_pose'].cuda(self.device)
        self.gt_rot = data['gt_local_rot'][..., 2:, :].cuda(self.device).view(-1, self.opt.num_rot_heatmap * 3)
        self.gt_limb_theta = data['gt_limb_theta'].cuda(self.device)
        
        self.gt_pelvis_left = data['gt_pelvis_left'].cuda(self.device)
        self.gt_pelvis_right =  data['gt_pelvis_right'].cuda(self.device)
        
        batch_dim = len(self.input_rgb_left.shape) - 3
        self.gt_pelvis = torch.stack((self.gt_pelvis_left, self.gt_pelvis_right), dim=batch_dim)

        self.gt_limb_heatmap_left = data['gt_limb_heatmap_left'].cuda(self.device)
        self.gt_limb_heatmap_right = data['gt_limb_heatmap_right'].cuda(self.device)
        self.gt_plength_left = data['gt_plength_left'].cuda(self.device)
        self.gt_plength_right = data['gt_plength_right'].cuda(self.device)
        
    def forward_heatmap(self):
        # estimate stereo heatmaps
        with torch.no_grad():
            pred_heatmap_cat = self.net_HeatMap(self.input_rgb_left, self.input_rgb_right)
            pred_limb_heatmap_cat = self.net_RotHeatMap(self.input_rgb_left, self.input_rgb_right)
            self.pred_heatmap_left, self.pred_heatmap_right = torch.chunk(pred_heatmap_cat, 2, dim=1)

            self.pred_limb_heatmap_left, self.pred_limb_heatmap_right = torch.chunk(pred_limb_heatmap_cat, 2, dim=1)
        
            pred_heatmap_cat = torch.cat((pred_heatmap_cat, pred_limb_heatmap_cat), dim=1)
        self.pred_heatmap_cat = pred_heatmap_cat
            
    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            # estimate pose and reconstruct stereo heatmaps
            self.forward_heatmap()
            
            self.pred_pose, self.pred_rot, pred_heatmap_rec_cat = self.net_AutoEncoder(self.pred_heatmap_cat, self.input_rgb_left, self.input_rgb_right)
            self.pred_heatmap_left_rec, self.pred_heatmap_right_rec = torch.chunk(pred_heatmap_rec_cat[:, :self.opt.num_heatmap*2], 2, dim=1)
            self.pred_limb_heatmap_left_rec, self.pred_limb_heatmap_right_rec = torch.chunk(pred_heatmap_rec_cat[:, self.opt.num_heatmap*2:], 2, dim=1)
    
    def backward_AutoEncoder(self):
        with autocast(enabled=self.opt.use_amp):
            gt_sqrt_limb_length_left = torch.sqrt(self.gt_plength_left[..., None, None])
            gt_sqrt_limb_length_right = torch.sqrt(self.gt_plength_right[..., None, None])
                
            losses = []
            
            loss_pose = self.lossfunc_MPJPE(self.pred_pose, self.gt_pose)
            loss_cos_sim = self.lossfunc_cos_sim(self.pred_pose, self.gt_pose)
            loss_heatmap_left_rec = self.lossfunc_MSE(
                self.pred_heatmap_left_rec, self.pred_heatmap_left.detach()
            )
            loss_heatmap_right_rec = self.lossfunc_MSE(
                self.pred_heatmap_right_rec, self.pred_heatmap_right.detach()
            )
            loss_heatmap_limb_left_rec = self.lossfunc_MSE(
                self.pred_limb_heatmap_left_rec / gt_sqrt_limb_length_left,
                self.pred_limb_heatmap_left.detach() / gt_sqrt_limb_length_left
            )
            loss_heatmap_limb_right_rec = self.lossfunc_MSE(
                self.pred_limb_heatmap_right_rec / gt_sqrt_limb_length_right,
                self.pred_limb_heatmap_right.detach() / gt_sqrt_limb_length_right
            )
            
            self.loss_pose = loss_pose * self.opt.lambda_mpjpe
            self.loss_cos_sim = loss_cos_sim * self.opt.lambda_cos_sim * self.opt.lambda_mpjpe
            self.loss_heatmap_left_rec = loss_heatmap_left_rec * self.opt.lambda_heatmap_rec
            self.loss_heatmap_right_rec = loss_heatmap_right_rec * self.opt.lambda_heatmap_rec
            self.loss_heatmap_limb_left_rec = loss_heatmap_limb_left_rec * self.opt.lambda_rot_heatmap_rec
            self.loss_heatmap_limb_right_rec = loss_heatmap_limb_right_rec * self.opt.lambda_rot_heatmap_rec
        
            losses.extend([self.loss_pose, self.loss_cos_sim,
                    self.loss_heatmap_left_rec, self.loss_heatmap_right_rec,
                    self.loss_heatmap_limb_left_rec, self.loss_heatmap_limb_right_rec])
            
            loss_rot = self.lossfunc_MSE(self.pred_rot, self.gt_rot)
            self.loss_rot = loss_rot * self.opt.lambda_rot
            losses.append(self.loss_rot)
                
            loss_total = sum(losses)

        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_AutoEncoder.train()
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_AutoEncoder()

        # optimizer step
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)

        self.scaler.update()
        
    def set_eval_mode(self):
        self.net_AutoEncoder.eval()
        self.net_HeatMap.eval()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.set_eval_mode()

        self.forward_heatmap()
        self.pred_pose, self.pred_rot, _ = self.net_AutoEncoder.forward(self.pred_heatmap_cat, self.input_rgb_left, self.input_rgb_right)

        S1_hat = batch_compute_similarity_transform_torch(self.pred_pose, self.gt_pose)

        # compute metrics
        for id in range(self.pred_pose.size()[0]):  # batch size
            # calculate mpjpe and p_mpjpe   # cm to mm
            mpjpe = self.lossfunc_MPJPE(self.pred_pose[id], self.gt_pose[id]) * self.cm2mm
            pa_mpjpe = self.lossfunc_MPJPE(S1_hat[id], self.gt_pose[id]) * self.cm2mm
                
            metrics = dict(
                mpjpe=mpjpe, 
                pa_mpjpe=pa_mpjpe)

            rot = self.lossfunc_MSE(self.pred_rot, self.gt_rot)
            metrics['rot'] = rot

            # update metrics dict
            runnning_average_dict.update(metrics)

        return self.pred_pose, self.pred_heatmap_cat, runnning_average_dict


