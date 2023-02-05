import torch
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from customdataset.get_load_dataset import *
from utils.fr_trainer import *

def train(args):
    DB = args.database
    flare_train_dataset_path = args.train_input_path
    flare_train_label_path = args.train_label_path
    flare_val_dataset_path = args.valid_input_path
    flare_val_label_path = args.valid_label_path
    weights_save_path = args.weights_save_path
    val_results_path = args.val_results_path
    FD_model_weights_path = args.fd_model_weights_path
    root = args.data_root
    save_root = args.save_root
    device = args.device
    patch_size = args.patch_size
    lr = args.lr
    NUM_EPOCH = args.num_epoch
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    flare_fold_train_dataset_path = sorted([os.path.join(root, flare_train_dataset_path, file) 
                                    for file in os.listdir(os.path.join(root, flare_train_dataset_path))])

    flare_fold_train_label_path = sorted([os.path.join(root, flare_train_label_path, file) 
                                for file in os.listdir(os.path.join(root, flare_train_label_path))])

    flare_fold_valid_dataset_path = sorted([os.path.join(root, flare_val_dataset_path, file) 
                                    for file in os.listdir(os.path.join(root, flare_val_dataset_path))])

    flare_fold_valid_label_path = sorted([os.path.join(root, flare_val_label_path, file) 
                                for file in os.listdir(os.path.join(root, flare_val_label_path))])

    if DB == 'CamVid':
        train_datasets = get_train_dataset_v2(
        inp_dir=flare_fold_train_dataset_path,
        tar_dir=flare_fold_train_label_path,
        )
        valid_datasets = get_val_test_dataset_v2(
        inp_dir=flare_fold_valid_dataset_path,
        tar_dir=flare_fold_valid_label_path,
        )
    elif DB == 'KITTI':
        train_datasets = get_Kitti_train_dataset_v2(
        inp_dir=flare_fold_train_dataset_path,
        tar_dir=flare_fold_train_label_path,
        )
        valid_datasets = get_Kitti_val_test_dataset_v2(
        inp_dir=flare_fold_valid_dataset_path,
        tar_dir=flare_fold_valid_label_path,
        )

    train_loader = DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    '''
    save_root, save_model_weights_path, save_val_results_path, FD_model_weights_path,  device, patch_size, lr, num_epochs
    '''
    trainer = Trainier(save_root, weights_save_path, val_results_path, FD_model_weights_path, device, patch_size, lr, NUM_EPOCH).to(device)
    trainer.train(train_loader, valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flare removal')
    parser.add_argument('--data_root', type=str, default='/workspace/Datas/')
    parser.add_argument('--database', type=str, default='CamVid')
    parser.add_argument('--train_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train')
    parser.add_argument('--train_label_path', type=str, default='camvid_label_12_2fold_v2/fold1_train/inputs')
    parser.add_argument('--valid_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid')
    parser.add_argument('--valid_label_path', type=str, default='camvid_label_12_2fold_v2/fold1_valid/inputs')
    parser.add_argument('--fd_model_weights_path', type=str, default='/workspace/DB1_proposed_model_fd_fold1_weights/model_168.pth')
    parser.add_argument('--weights_save_path', type=str, default='DB1_proposed_cam_frn_fold1_weights-2_fold')
    parser.add_argument('--val_results_path', type=str, default='DB1_proposed_cam_frn_fold1_val_results-2_fold')
    parser.add_argument('--save_root', type=str, default='/workspace/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    args = parser.parse_args()

    train(args)
