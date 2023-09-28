import os
import numpy as np
from scipy.ndimage import gaussian_filter
from utils.projection import world2cam
from utils.util import get_index_to_joint_name, get_kinematic_parents

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_npy_path(opt, npy_path):
    npy_subpath = npy_path.replace(opt.data_dir, "", 1).replace(opt.data_sub_path, "", 1).replace(".npy", "", 1)
    npy_name = npy_subpath.replace("/", "-").replace("\\", "-").replace(".", "-")

    [head, tail] = os.path.split(npy_path)
    take_data_dir = os.path.join(head, os.pardir)

    json_path = os.path.join(os.path.join(take_data_dir, 'json'), tail[:-4] + ".json")

    return npy_subpath, npy_name, head, tail, take_data_dir, json_path

def get_num_joints(opt):
    joint_preset = opt.joint_preset
    index_to_joint_name = get_index_to_joint_name(joint_preset)
    num_joints = len(index_to_joint_name)
    return num_joints

def get_local_rot(opt, pose3d):
    num_joints = get_num_joints(opt)
    joint_orient = np.zeros(shape=(num_joints, 3), dtype=np.float32)
    for i in range(1, num_joints):
        joint_pos_delta = np.array(pose3d[i]) - np.array(pose3d[get_kinematic_parents(opt.joint_preset)[i]])
        joint_orient[i] = joint_pos_delta / np.linalg.norm(joint_pos_delta, axis=-1)
    return joint_orient

def pts2d_to_heatmap(coord, res=64, area=64):
    heatmap = np.zeros((area, area), dtype=np.float32)
    if len(coord.shape) == 1:
        coord = coord[None, :]
    
    coords = coord
    
    for coord in coords:
        hm_coord = np.rint(coord/(1024.0 / res)).astype(int)
        padding = (area - res) // 2
        hm_coord += padding

        if 0 <= hm_coord[0] < area and 0 <= hm_coord[1] < area:
            heatmap[hm_coord[1], hm_coord[0]] = 1
            heatmap = gaussian_filter(heatmap, sigma=1)
            heatmap /= 0.15915589174187972
    
    return heatmap

def _fpart(x):
    return x - int(x)

def _rfpart(x):
    return 1 - _fpart(x)


from skimage.draw import line_aa
def get_line_limb_heatmap(p_coord, coord, limb_heatmap=None, res=64):
    if limb_heatmap is None:
        limb_heatmap = np.zeros((res, res))
    p_coord = np.rint(p_coord).astype(int)
    coord = np.rint(coord).astype(int)
    rr, cc, val = line_aa(p_coord[0], p_coord[1], coord[0], coord[1])
    
    idx = np.logical_and(np.logical_and(rr >= 0, rr <= res-1), np.logical_and(cc >= 0, cc <= res-1))
    limb_heatmap[cc[idx], rr[idx]] = val[idx]
    
    return limb_heatmap


def get_points_limb_heatmap(p_coord, coord, limb_heatmap=None, res=64, area=64):
    if limb_heatmap is None:
        limb_heatmap = np.zeros((area, area))
    heatmap = pts2d_to_heatmap(np.stack((p_coord, coord)), res, area)
    limb_heatmap += heatmap
        
    return limb_heatmap
        
        
def get_limb_data(pts2d, pts3d, res=64, area=None, htype='line', sigma=1, joint_preset="UnrealEgo"):
    index_to_joint_name = get_index_to_joint_name(joint_preset)
    kinematic_parents = get_kinematic_parents(joint_preset)
    num_joints = len(index_to_joint_name)
    
    if area is None:
        area = res
    limb_heatmaps = np.zeros((num_joints - 2, area, area), dtype=np.float32)
    lengths = np.zeros(num_joints - 2, dtype=np.float32)
    square_sums = np.zeros(num_joints - 2, dtype=np.float32)
    theta = np.zeros(num_joints - 2, dtype=np.float32)
    
    if (area - res) % 2 != 0:
        print('area - res must be even number')
        exit()
    padding = (area - res) // 2
    
    for joint_idx in range(2, num_joints):
        assign_idx = joint_idx - 2
        parent_idx = kinematic_parents[joint_idx]
        
        divider = (1024.0 / res)
        p_coord = pts2d[parent_idx]
        coord = pts2d[joint_idx]
        p_coord = p_coord/divider
        coord = coord/divider
        p_coord3d = pts3d[parent_idx]
        coord3d = pts3d[joint_idx]
        
        # sign = p_coord3d[2] > coord3d[2]
        limb_3d = p_coord3d - coord3d
        limb_2dlen = np.linalg.norm(limb_3d[:2])
        theta[assign_idx] = np.arctan(limb_3d[2]/limb_2dlen)
        
        limb_heatmap = np.zeros((res, res), dtype=np.float32)
        limb_pixel_length = np.linalg.norm(p_coord - coord) + 1.0
    
        p_coord += padding
        coord += padding
        
        if htype == 'line':
            lengths[assign_idx] = limb_pixel_length
            limb_heatmap = get_line_limb_heatmap(p_coord, coord, limb_heatmap, res)
        elif htype == 'points':
            lengths[assign_idx] = 2
            limb_heatmap = get_points_limb_heatmap(p_coord, coord, limb_heatmap, res)
        else:
            raise Exception("Undefined limb heatmap type")
            
        square_sums[assign_idx] += np.sum(np.square(limb_heatmap))
        
        limb_heatmap = gaussian_filter(limb_heatmap, sigma=sigma, mode='constant')
        limb_heatmap *= sigma
        
        limb_heatmaps[assign_idx] = limb_heatmap
        
        if lengths[assign_idx] < limb_pixel_length - 2:
            print("Curve is shorter than a line. {}<{}-1".format(lengths[assign_idx], limb_pixel_length))

    return limb_heatmaps, lengths, square_sums, theta

def overwrite_limb_data(npy_item, pts2d_left, pts2d_right, pts3d_left, pts3d_right,
                        res=64, area=64, htype='line', sigma=1, joint_preset=None):
    (npy_item['gt_limb_heatmap_left'],
        npy_item['gt_pixel_length_left'],
        npy_item['gt_sqsum_limb_heatmap_left'],
        npy_item['gt_limb_theta'],) = get_limb_data(pts2d_left, pts3d_left, res, area, htype, sigma=sigma, joint_preset=joint_preset)
    (npy_item['gt_limb_heatmap_right'], \
        npy_item['gt_pixel_length_right'],
        npy_item['gt_sqsum_rot_heatmap_right'],
        _, ) = get_limb_data(pts2d_right, pts3d_right, res, area, htype, sigma=sigma, joint_preset=joint_preset)
