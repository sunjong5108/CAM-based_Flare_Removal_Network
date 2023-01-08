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
from utils.DB2_trainer import Trainier

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    root = '/workspace/Datas/'

    Kitti_flare_fold2_train_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold2_train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold2_train'))])

    Kitti_flare_fold2_train_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold2_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold2_train', 'inputs'))])

    Kitti_flare_fold2_train_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'fold2_train', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'fold2_train'))])

    Kitti_flare_fold2_val_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold2_val', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'fold2_val'))])

    Kitti_flare_fold2_val_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'fold2_val', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'fold2_val', 'inputs'))])

    Kitti_flare_fold2_val_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'fold2_val', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'fold2_val'))])              

    Kitti_flare_fold2_test_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'train'))])

    Kitti_flare_fold2_test_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'train', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'train', 'inputs'))])

    Kitti_flare_fold2_test_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'train', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'train'))])
    
    NUM_EPOCH = 400
    lr = 1e-4
    patch_size = 300

    train_datasets = get_Kitti_train_dataset(
        inp_dir=Kitti_flare_fold2_train_dataset_path,
        tar_dir=Kitti_flare_fold2_train_label_path,
        flare_dir=Kitti_flare_fold2_train_flare_path
    )
    train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, drop_last=False)

    valid_datasets = get_Kitti_val_test_dataset(
        inp_dir=Kitti_flare_fold2_val_dataset_path,
        tar_dir=Kitti_flare_fold2_val_label_path,
        flare_dir=Kitti_flare_fold2_val_flare_path
    )
    valid_loader = DataLoader(valid_datasets, batch_size=1, shuffle=False, drop_last=False)

    trainer = Trainier('DB2_proposed_model_fold2_weights_with_CAM_tv', 'DB2_proposed_model_fold2_val_results_with_CAM_tv', 
                    'fold2', 'cuda', patch_size, lr, NUM_EPOCH).to('cuda')
    trainer.train(train_loader, valid_loader)
# Best PSNR score index: 349 Best PSNR score: 26.682926765047718
# Best SSIM score index: 392 Best SSIM score: 0.9040866608297468
# Best FID score index: 382 Best FID score: 34.14624293392166

# Best PSNR score index: 387 Best PSNR score: 26.777212909224616
# Best SSIM score index: 390 Best SSIM score: 0.9067601156380269
# Best FID score index: 356 Best FID score: 33.846908888656344