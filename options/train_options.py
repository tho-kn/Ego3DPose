from .dataset_options import DatasetOptions

class TrainOptions(DatasetOptions):
    def initialize(self):
        DatasetOptions.initialize(self)

        # ------------------------------ training epoch ------------------------------ #
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count')
        self.parser.add_argument('--niter', type=int, default=0,
                                 help='# of iter with initial learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0,
                                 help='# of iter to decay learning rate to zero')


        # ------------------------------ learning rate and loss weight ------------------------------ #
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy[lambda|step|plateau]')
        self.parser.add_argument('--lr_decay_iters_step', type=int, default=4,
                                 help='of iter to decay learning rate with a policy [step]')
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='initial learning rate for optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.0,
                                 help='weight decay')

        self.parser.add_argument('--lambda_mpjpe', type=float, default=1.0,
                                 help='weight for loss_mpjpe')
        self.parser.add_argument('--lambda_pelvis', type=float, default=0.01,
                                 help='weight for loss_pelvis')
        self.parser.add_argument('--lambda_rot', type=float, default=1.0,
                                 help='weight for loss_rot')
        self.parser.add_argument('--lambda_heatmap', type=float, default=1.0,
                                 help='weight for loss_heatmap')
        self.parser.add_argument('--lambda_segmentation', type=float, default=1.0,
                                 help='weight for loss_segmentation')
        self.parser.add_argument('--lambda_rot_heatmap', type=float, default=1.0,
                                 help='weight for loss_rot_heatmap')
        self.parser.add_argument('--lambda_heatmap_rec', type=float, default=1e-3,
                                 help='weight for loss_heatmap_rec')
        self.parser.add_argument('--lambda_rot_heatmap_rec', type=float, default=1e-3,
                                 help='weight for loss_rot_heatmap_rec')
        self.parser.add_argument('--lambda_cos_sim', type=float, default=-1e-2,
                                 help='weight for loss_cos_sim')

        # ------------------------------ display the results ------------------------------ #
        self.parser.add_argument('--display_freq', type=int, default=1,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_epoch_freq', type=int, default=1,
                                 help='frequency of showing training results at the end of epochs')
        self.parser.add_argument('--save_latest_freq', type=int, default=1,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--val_epoch_freq', type=int, default=1,
                                 help='frequency of validation')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        
        # ------------------------------ others ------------------------------ #
        self.parser.add_argument('--gradient_checkpoint', action='store_true',
                                 help='use gradient checkpointing to save memory')

        self.isTrain = True
