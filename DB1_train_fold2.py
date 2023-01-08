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
from utils.DB1_trainer import Trainier

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    root = '/workspace/Datas/'

    Camvid_flare_fold2_train_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold2_train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold2_train'))])

    Camvid_flare_fold2_train_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'inputs'))])

    Camvid_flare_fold2_train_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold2_train', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold2_train'))])

    Camvid_flare_fold2_val_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold2_val', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold2_val'))])

    Camvid_flare_fold2_val_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'inputs'))])

    Camvid_flare_fold2_val_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold2_val', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold2_val'))])              

    Camvid_flare_fold2_test_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'train'))])

    Camvid_flare_fold2_test_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs'))])

    Camvid_flare_fold2_test_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'train', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'train'))])
    
    NUM_EPOCH = 400
    lr = 1e-4
    patch_size = 300

    train_datasets = get_train_dataset(
        inp_dir=Camvid_flare_fold2_train_dataset_path,
        tar_dir=Camvid_flare_fold2_train_label_path,
        flare_dir=Camvid_flare_fold2_train_flare_path
    )
    train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, drop_last=False)

    valid_datasets = get_val_test_dataset(
        inp_dir=Camvid_flare_fold2_val_dataset_path,
        tar_dir=Camvid_flare_fold2_val_label_path,
        flare_dir=Camvid_flare_fold2_val_flare_path
    )
    valid_loader = DataLoader(valid_datasets, batch_size=1, shuffle=False, drop_last=False)

    test_datasets = get_val_test_dataset(
        inp_dir=Camvid_flare_fold2_test_dataset_path,
        tar_dir=Camvid_flare_fold2_test_label_path,
        flare_dir=Camvid_flare_fold2_test_flare_path
    )
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, drop_last=False)

    trainer = Trainier('DB1_proposed_model_fold2_weights_with_CAM_tv', 'DB1_proposed_model_fold2_val_results_with_CAM_tv', 
                    'fold2', 'cuda', patch_size, lr, NUM_EPOCH).to('cuda')
    trainer.train(train_loader, valid_loader)
# Best PSNR score index: 400 Best PSNR score: 28.08770353968149
# Best SSIM score index: 400 Best SSIM score: 0.9303313181133064
# Best FID score index: 348 Best FID score: 29.66964323902253