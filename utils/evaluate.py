from .util import RunningAverageDict
from tqdm import tqdm
import torch
import os
import numpy as np
from .util import batch_compute_similarity_transform_torch
from .loss import LossFuncMPJPE

def get_save_path(opt):
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, opt.result_dir)
    save_path = os.path.join(save_path, opt.experiment_name)
    return save_path
        
def get_dict_motion_category():
    return {
        "001": "jumping",
        "002": "falling_down",
        "003": "exercising",
        "004": "pulling",
        "005": "singing",
        "006": "rolling",
        "007": "crawling",
        "008": "laying",
        "009": "sitting_on_the_ground",
        "010": "crouching",
        "011": "crouching_and_tuning",
        "012": "crouching_to_standing",
        "013": "crouching_and_moving_forward",
        "014": "crouching_and_moving_backward",
        "015": "crouching_and_moving_sideways",
        "016": "standing_with_whole_body_movement",
        "017": "standing_with_upper_body_movement",
        "018": "standing_and_turning",
        "019": "standing_to_crouching",
        "020": "standing_and_moving_forward",
        "021": "standing_and_moving_backward",
        "022": "standing_and_moving_sideways",
        "023": "dancing",
        "024": "boxing",
        "025": "wrestling",
        "026": "soccer",
        "027": "baseball",
        "028": "basketball",
        "029": "american_football",
        "030": "golf",
    }

lossfunc_MPJPE = LossFuncMPJPE()
cm2mm = 10

def compute_metrics(pred_pose, gt_pose, running_average_dict):
    S1_hat = batch_compute_similarity_transform_torch(pred_pose, gt_pose)

    mpjpes = torch.zeros(pred_pose.size()[0])
    pa_mpjpes = torch.zeros_like(mpjpes)

    # compute metrics
    for id in range(pred_pose.size()[0]):  # batch size
        mpjpe = lossfunc_MPJPE(pred_pose[id], gt_pose[id]) * cm2mm
        pa_mpjpe = lossfunc_MPJPE(S1_hat[id], gt_pose[id]) * cm2mm
        
        # update metrics dict
        running_average_dict.update(dict(
            mpjpe=mpjpe,
            pa_mpjpe=pa_mpjpe)
        )
        mpjpes[id] = mpjpe
        pa_mpjpes[id] = pa_mpjpe
        
    return mpjpes, pa_mpjpes
    
def test_evaluate(opt, model, eval_dataset, epoch, save_result=False):
    running_average_dict = RunningAverageDict()
    running_average_dict_dummy = RunningAverageDict()

    if opt.use_slurm is False:
        bar_eval = tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc=f"Epoch: {epoch}", position=0, leave=True)
    else:
        bar_eval = enumerate(eval_dataset) 
    
    per_frame_running_average_dict = []
    
    stats = {"mpjpe":[], "pa_mpjpe":[]}

    if len(eval_dataset) == 0:
        running_average_dict.update({})
        print("Evaluation dataset is empty!")
        return running_average_dict.get_value(), list(map(lambda x: x.get_value(), per_frame_running_average_dict)), stats
        
    model.eval()
    model.set_eval_mode()
    
    pred_poses = []
    gt_poses = []
    input_paths = []
    
    with torch.no_grad():
        for id, data in bar_eval:
            model.set_input(data)
            if save_result:
                input_paths.append(data["frame_data_path"])
            
            pred_pose, _, running_average_dict_dummy = model.evaluate(runnning_average_dict=running_average_dict_dummy)
            
            pred_pose = model.pred_pose
            gt_pose = model.gt_pose
    
            if save_result:
                pred_poses.append(pred_pose.cpu().numpy())
                gt_poses.append(gt_pose.cpu().numpy())
                    
            # compute metrics
            mpjpes, pa_mpjpes = compute_metrics(pred_pose, gt_pose, running_average_dict)
            stats["mpjpe"].extend(mpjpes)
            stats["pa_mpjpe"].extend(pa_mpjpes)

    model.train()
    
    if save_result:
        pred_pose = np.concatenate(pred_poses, axis=0)
        gt_pose = np.concatenate(gt_poses, axis=0)
        input_paths = np.array(input_paths, dtype=object)
        
        save_path = get_save_path(opt)
        np.save(os.path.join(save_path, "pred_pose.npy"), pred_pose)
        np.save(os.path.join(save_path, os.pardir, "gt_pose.npy"), gt_pose)
        np.save(os.path.join(save_path, os.pardir, "input_paths.npy"), input_paths)
        
        import pickle
        pickle.dump(input_paths, open(os.path.join(save_path, "input_paths.pkl"), "wb"))
    
    return running_average_dict.get_value(), list(map(lambda x: x.get_value(), per_frame_running_average_dict)), stats
    
def train_evaluate(opt, model, eval_dataset, epoch):
    model.eval()
    runnning_average_dict = RunningAverageDict()

    if opt.use_slurm is False:
        bar_eval = tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc=f"Epoch: {epoch}", position=0, leave=True)
    else:
        bar_eval = enumerate(eval_dataset) 
    
    if len(eval_dataset) == 0:
        runnning_average_dict.update({})
        print("Evaluation dataset is empty!")

    with torch.no_grad():
        for id, data in bar_eval:
            torch.cuda.empty_cache()
            model.set_input(data)
            pred, pred_heatmap, runnning_average_dict = model.evaluate(runnning_average_dict=runnning_average_dict)
                
    model.train()

    return runnning_average_dict.get_value()

