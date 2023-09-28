from .base_options import BaseOptions

class DatasetOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--default_data_path', type=str, default="./UnrealEgoData",
                                help='default path to the UnrealEgo dataset written in data list')
        self.parser.add_argument('--data_dir', type=str, default="/ssd_data1/UnrealEgoData",
                                help='path to the UnrealEgo dataset')
        self.parser.add_argument('--data_sub_path', type=str, default='all_data_with_img-256_hm-64_pose-16_npy',
                                help='sub path to npy files')
        self.parser.add_argument('--metadata_dir', nargs='+', type=str,
                                default=["/ssd_data1/UnrealEgoData"],
                                help='default path to the UnrealEgo dataset metadata files, it can be splited to multiple folders.')
        self.parser.add_argument('--data_prefix', type=str, default="",
                                help='prefix to the UnrealEgo dataset list files')
        self.parser.add_argument('--joint_preset', type=str, default="UnrealEgo",
                                 help='preset for joint order and parents')