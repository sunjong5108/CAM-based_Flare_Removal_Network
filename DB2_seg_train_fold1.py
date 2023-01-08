import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from customdataset.get_load_dataset import *
from utils.SegTrainer import *

if __name__ == "__main__":
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

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    root = '/workspace/Datas/'

    Kitti_flare_fold1_train_input_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'inputs'))])
    Kitti_flare_fold1_train_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'labels'))])

    Kitti_flare_fold1_val_input_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'inputs'))])
    Kitti_flare_fold1_val_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'labels', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'labels'))])

    Kitti_flare_fold1_test_input_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs'))])
    Kitti_flare_fold1_test_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'test', 'labels', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'test', 'labels'))])

    
    NUM_EPOCH = 200
    lr = 1e-4
    batch_size = 4
    eval_batch = 1
    patch_size = 300
    
    train_datasets = SEG_Kitti_get_train_dataset(
        inp_dir=Kitti_flare_fold1_train_input_path,
        tar_dir=Kitti_flare_fold1_train_label_path,
        patch_size = patch_size
    )
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    valid_datasets = SEG_Kitti_get_val_test_dataset(
        inp_dir=Kitti_flare_fold1_val_input_path,
        tar_dir=Kitti_flare_fold1_val_label_path
    )
    valid_loader = DataLoader(valid_datasets, batch_size=eval_batch, shuffle=False, drop_last=False)

    test_datasets = SEG_Kitti_get_val_test_dataset(
        inp_dir=Kitti_flare_fold1_test_input_path,
        tar_dir=Kitti_flare_fold1_test_label_path
    )
    test_loader = DataLoader(test_datasets, batch_size=eval_batch, shuffle=False, drop_last=False)

    #train_loader, label_color_map, save_model_weights_path, device, lr, num_epochs
    trainer = Trainier(train_loader, label_color_map, 'DB2_proposed_segmodel_fold1_weights', 'cuda', lr, NUM_EPOCH).to('cuda')
    trainer.train(train_loader, valid_loader, test_loader, batch_size, eval_batch)
# Validation
# Best MIOU score index: 129 Best PSNR score: 0.6648778789213254
# Test
# Best MIOU score index: 189 Best PSNR score: 0.6613696053391894
    