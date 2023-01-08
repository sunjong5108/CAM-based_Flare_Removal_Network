import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from customdataset.get_load_dataset import *
from losses.losses import *
from models.proposed_model import *
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from utils.trainer_flare_detect import *

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    root = '/workspace/Datas/'

    Kitti_flare_fold1_train_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold1_train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold1_train'))])

    Kitti_flare_fold1_train_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_train', 'inputs'))])

    Kitti_flare_fold1_train_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'fold1_train', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'fold1_train'))])

    Kitti_flare_fold1_val_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold1_val', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold1_val'))])

    Kitti_flare_fold1_val_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold1_val', 'inputs'))])

    Kitti_flare_fold1_val_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'fold1_val', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'fold1_val'))])              

    Kitti_flare_fold1_test_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'test', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'test'))])

    Kitti_flare_fold1_test_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs'))])

    Kitti_flare_fold1_test_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'test', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'test'))])

    NUM_EPOCH = 200
    lr =3e-5

    train_datasets = FD_Kitti_get_train_dataset(
        inp_dir=Kitti_flare_fold1_train_dataset_path + Kitti_flare_fold1_train_label_path
    )
    train_loader = DataLoader(train_datasets, batch_size=4, shuffle=True, drop_last=False)

    valid_datasets = FD_Kitti_get_val_test_dataset(
        inp_dir=Kitti_flare_fold1_val_dataset_path + Kitti_flare_fold1_val_label_path
    )
    valid_loader = DataLoader(valid_datasets, batch_size=1, shuffle=False, drop_last=False)

    trainer = Trainier('DB2_proposed_model_fd_fold1_weights', 'cuda', lr, NUM_EPOCH).to('cuda')
    trainer.train(train_loader, valid_loader)
   #Best loss index: 195 Best loss:  0.00016423793290827731