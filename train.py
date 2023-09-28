from tqdm import tqdm
import numpy as np
import os, shutil
from options.train_options import TrainOptions
from dataloader.data_loader import  dataloader_full
from model.models import create_model
from utils.evaluate import get_dict_motion_category, train_evaluate
from utils.util import print_current_errors
from utils.util import RunningAverageDict
from torch.utils.tensorboard import SummaryWriter
import  math

def prepare_summary(opt, clear_summary=False, purge_step=None):
    summary_dir = os.path.join(opt.log_dir, opt.experiment_name, 'summary')
    
    if clear_summary:
        if os.path.exists(summary_dir) and os.path.isdir(summary_dir):
            test_result = os.path.join(opt.log_dir, opt.experiment_name, 'test_result.txt')
            if os.path.exists(test_result):
                old_summary_idx = 0
                old_summary_dir = summary_dir + '_' + str(old_summary_idx)
                while os.path.exists(old_summary_dir) and os.path.isdir(old_summary_dir):
                    old_summary_idx += 1
                    old_summary_dir = summary_dir + '_' + str(old_summary_idx)
                shutil.move(summary_dir, old_summary_dir)
                test_result = os.path.join(opt.log_dir, opt.experiment_name, 'test_result.txt')
                old_test_result = test_result[:-4] + '_' + str(old_summary_idx) + ".txt"
                shutil.move(test_result, old_test_result)
            else:
                shutil.rmtree(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir, purge_step=purge_step)
    
    return writer

def record_dataset_information():
    dataset_log_dir = os.path.join(opt.log_dir, opt.experiment_name, 'dataset')
    if os.path.exists(dataset_log_dir) and os.path.isdir(dataset_log_dir):
        shutil.rmtree(dataset_log_dir)
        
    os.makedirs(dataset_log_dir, exist_ok=True)
    mod_dataset_path = os.path.join(opt.data_dir, "modify_dataset_log.txt")
    if os.path.exists(mod_dataset_path):
        shutil.copy(mod_dataset_path, os.path.join(dataset_log_dir, "modify_dataset_log.txt"))
    script_path = os.path.join(opt.data_dir, "script.py")
    if os.path.exists(script_path):
        shutil.copy(script_path, os.path.join(dataset_log_dir, "script.py"))
        
def test_model(opt, model):
    test_dataset = dataloader_full(opt, mode='test')
    print('test images = {}'.format(len(test_dataset) * opt.batch_size))
    
    print("\n")
    print("load best model ...")
    metrics_test =  train_evaluate(opt, model, test_dataset, "best")
    
    print("best test metrics:")
    for k, v in metrics_test.items():
        print("{}: {}".format(k, v))
        
    return metrics_test

def train_main(opt, checkpoint_dir=None):
    
    print("preparing dataset ... ")
    train_dataset = dataloader_full(opt, mode='train')
    val_dataset = dataloader_full(opt, mode='validation')

    print('train images = {}'.format(len(train_dataset) * opt.batch_size))
    print('validation images = {}'.format(len(val_dataset) * opt.batch_size))
    
    model = create_model(opt)
    # model = torch.compile(model)

    total_steps=0
    current_best_metrics = np.inf
    best_metrics = None
    
    writer = prepare_summary(opt, clear_summary=(opt.epoch_count==1))
    record_dataset_information()
    
    print('---------------------Start Training-----------------------')
    model.train()
    
    if checkpoint_dir is not None:
        model.load_networks(checkpoint_path=checkpoint_dir)
    
    if opt.epoch_count > 1:
        model.load_networks(which_epoch=opt.epoch_count-1)
        
    loss_records = {}
    auto_restart = True
    if opt.model == "egoglass":
        auto_restart = False
    if opt.epoch_count > 1:
        auto_restart = False
        
    total_itr = 0
    
    for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
        epoch_iter = 0
        print('-----------------Train Epoch: {}-----------------'.format(str(epoch)))
        
        curr_loss = {}
        
        if not opt.use_slurm:
            bar_train = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch: {epoch}", position=0, leave=True, dynamic_ncols=True)
        else:
            bar_train = enumerate(train_dataset) 
        total_loss = RunningAverageDict()

        # training
        for i, data in bar_train:
            total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            
            curr_itr = total_itr + i
            total_loss.update(model.get_current_errors())
            for k, v in model.get_current_errors().items():
                if auto_restart and curr_itr < 2000 and (k.startswith('heatmap') or k.startswith('limb_heatmap')) and 'rec' not in k:
                    if k not in loss_records:
                        loss_records[k] = (curr_itr, v)
                    else:
                        if v < loss_records[k][1]:
                            loss_records[k] = (curr_itr, v)
                        else:
                            if curr_itr - loss_records[k][0] > 300:
                                print("Early heatmap convergence detected at: {} at {}!".format(i, v))
                                print("It leads to suboptimal results. Retrying..")
                                return False
                writer.add_scalar(f"Batch/{k}", v, i + len(train_dataset) * epoch)
            curr_loss = list(model.get_current_errors().values())
            curr_loss = ''.join(['%.3E ' % v for v in curr_loss])
            if not opt.use_slurm:
                bar_train.set_description(f"Epoch: {epoch}, Error: {curr_loss}")
            data = None
            
        if (epoch % opt.val_epoch_freq == 0):
            print('-----------------Validation Epoch: {}-----------------'.format(str(epoch)))
            metrics = train_evaluate(opt, model, val_dataset, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f"Validation/{k}", v, epoch)
            
            metric_string = ' '.join(['%s: %.3E' % (k, v) for k, v in metrics.items()])
            print(metric_string)

            if metrics['{}'.format(model.eval_key)] < current_best_metrics:
                current_best_metrics = metrics['{}'.format(model.eval_key)]
                model.save_networks('best')
                best_metrics = metrics

        if epoch % opt.print_epoch_freq == 0:
            print_current_errors(epoch, epoch_iter, total_loss.get_value(), epoch)
            for k, v in total_loss.get_value().items():
                writer.add_scalar(f"Train/{k}", v, epoch)
        
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        model.update_learning_rate()
        total_itr += len(bar_train)

        print('dir name: {}'.format(opt.experiment_name)) 

    print("\n")
    print("train finished !!!")
        
    writer.close()
    
    print("\n")
    print("best validation metrics: {}".format(best_metrics))
    print("\n")

    print('-----------------Test Best Model-----------------')
    model.load_networks("best")
    metrics_test = test_model(opt, model)
        
    print("\n")
    print("test finished !!!")
    print("\n")

    test_result_path = os.path.join(opt.log_dir, opt.experiment_name, "test_result.txt")
    test_result_file = open(test_result_path, "w")
        
    for k, v in metrics_test.items():
        test_result_file.write("{}: {}".format(k, v))
    print("\n")

    print('-----------------Start Category-Specific Evaluation-----------------')
    print("\n")

    def print_and_write(string):
        test_result_file.write(string + "\n")
        
    dict_motion_category = get_dict_motion_category()
    for key, value in dict_motion_category.items():
        key_test_dataset = dataloader_full(opt, mode="test", id=key)
        if len(key_test_dataset) == 0:
            print("{}:{} Test Dataset is Empty!".format(key, value))
            continue
        key_metrics_test = train_evaluate(opt, model, key_test_dataset, "best_" + key)
        print_and_write("category: {}".format(key + "_" + value))
        print_and_write("number of batches: {}".format(len(key_test_dataset)))
        for k, v in key_metrics_test.items():
            print_and_write("{}: {}".format(k, v))
        print("\n")

    print('-----------------All Process Finished-----------------')
    print("\n")
    
    return True
    
if __name__ == '__main__':
    opt = TrainOptions().parse()
    while True:
        result = train_main(opt)
        if result:
            break
