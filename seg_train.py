import argparse
import os
import torch
import numpy as np
from customdataset.get_load_dataset import *
from utils.SegTrainer_v3 import *

def train(args):
    DB = args.database
    train_dataset_path = args.train_input_path
    train_label_path = args.train_label_path
    val_dataset_path = args.valid_input_path
    val_label_path = args.valid_label_path
    weights_save_path = args.weights_save_path
    device = args.device
    root = args.data_root
    save_root = args.save_root
    lr = args.lr
    NUM_EPOCH = args.num_epoch
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    label_color_map = {"Sky": [128, 128, 128],
                    "Building": [128, 0, 0],
                    "Pole": [192, 192, 128],
                    "Road": [128, 64, 128],
                    "Pavement": [0, 0, 192],
                    "Tree": [128, 128, 0],
                    "SignSymbol": [192, 128, 128],
                    "Fence": [64, 64, 128],
                    "Car": [64, 0, 128],
                    "Pedestrian": [64, 64, 0],
                    "Bicyclist": [0, 128, 192],
                    "Void": [0, 0, 0]}

    fold_train_dataset_path = sorted([os.path.join(root, train_dataset_path, file) 
                                    for file in os.listdir(os.path.join(root, train_dataset_path))])

    fold_train_label_path = sorted([os.path.join(root, train_label_path, file) 
                                for file in os.listdir(os.path.join(root, train_label_path))])

    fold_valid_dataset_path = sorted([os.path.join(root, val_dataset_path, file) 
                                    for file in os.listdir(os.path.join(root, val_dataset_path))])

    fold_valid_label_path = sorted([os.path.join(root, val_label_path, file) 
                                for file in os.listdir(os.path.join(root, val_label_path))])

    if DB == 'CamVid':
        train_datasets = SEG_get_train_dataset(
        inp_dir=fold_train_dataset_path,
        tar_dir=fold_train_label_path,
        patch_size=args.patch_size
        )
        valid_datasets = SEG_get_val_test_dataset(
        inp_dir=fold_valid_dataset_path,
        tar_dir=fold_valid_label_path,
        )
    elif DB == 'KITTI':
        train_datasets = SEG_Kitti_get_train_dataset(
        inp_dir=fold_train_dataset_path,
        tar_dir=fold_train_label_path,
        patch_size=args.patch_size
        )
        valid_datasets = SEG_Kitti_get_val_test_dataset(
        inp_dir=fold_valid_dataset_path,
        tar_dir=fold_valid_label_path,
        )

    train_loader = DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    '''
    train_loader, label_color_map, save_root, save_model_weights_path, device, lr, num_epochs
    '''
    trainer = Trainier(train_loader, label_color_map, save_root, weights_save_path, device, lr, NUM_EPOCH).to(device)
    trainer.train(train_loader, valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DeepLabV3 plus')
    parser.add_argument('--data_root', type=str, default='/workspace/Datas/')
    parser.add_argument('--database', type=str, default='CamVid')
    parser.add_argument('--train_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train')
    parser.add_argument('--train_label_path', type=str, default='camvid_label_12_2fold_v2/fold1_train/inputs')
    parser.add_argument('--valid_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid')
    parser.add_argument('--valid_label_path', type=str, default='camvid_label_12_2fold_v2/fold1_valid/inputs')
    parser.add_argument('--weights_save_path', type=str, default='DB1_proposed_cam_frn_fold1_weights-2_fold')
    parser.add_argument('--save_root', type=str, default='/workspace/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    args = parser.parse_args()

    train(args)
