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

    Camvid_flare_fold1_train_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold1_train', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold1_train'))])

    Camvid_flare_fold1_train_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'inputs'))])

    Camvid_flare_fold1_train_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold1_train', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold1_train'))])

    Camvid_flare_fold1_val_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold1_val', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'fold1_val'))])

    Camvid_flare_fold1_val_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'inputs'))])

    Camvid_flare_fold1_val_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold1_val', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'fold1_val'))])              

    Camvid_flare_fold1_test_dataset_path = sorted([os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'test', file) 
                                    for file in os.listdir(os.path.join(root, 'flare_synthesized_CamVid_12label_Dataset_0527', 'test'))])

    Camvid_flare_fold1_test_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs', file) 
                                    for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs'))])

    Camvid_flare_fold1_test_flare_path = sorted([os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'test', file) 
                                for file in os.listdir(os.path.join(root, 'flare srgb CamVid 12label Dataset_0527', 'test'))])
    
    NUM_EPOCH = 400
    lr = 1e-4
    patch_size = 300

    train_datasets = get_train_dataset(
        inp_dir=Camvid_flare_fold1_train_dataset_path,
        tar_dir=Camvid_flare_fold1_train_label_path,
        flare_dir=Camvid_flare_fold1_train_flare_path
    )
    train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, drop_last=False)

    valid_datasets = get_val_test_dataset(
        inp_dir=Camvid_flare_fold1_val_dataset_path,
        tar_dir=Camvid_flare_fold1_val_label_path,
        flare_dir=Camvid_flare_fold1_val_flare_path
    )
    valid_loader = DataLoader(valid_datasets, batch_size=1, shuffle=False, drop_last=False)

    test_datasets = get_val_test_dataset(
        inp_dir=Camvid_flare_fold1_test_dataset_path,
        tar_dir=Camvid_flare_fold1_test_label_path,
        flare_dir=Camvid_flare_fold1_test_flare_path
    )
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, drop_last=False)

    #dest_model_path = os.path.join('/workspace/proposed_model_stage1_fold1_weights/', 'model_381.pth')
    trainer = Trainier('DB1_proposed_model_fold1_weights_with_CAM_tv', 'DB1_proposed_model_fold1_val_results_with_CAM_tv', 
                        'fold1', 'cuda', patch_size, lr, NUM_EPOCH).to('cuda')
    trainer.train(train_loader, valid_loader)
# Best PSNR score index: 384 Best PSNR score: 27.55210623618183
# Best SSIM score index: 385 Best SSIM score: 0.9280038169584882
# Best FID score index: 398 Best FID score: 41.01157670888634

# Best PSNR score index: 376 Best PSNR score: 27.386820529531768
# Best SSIM score index: 353 Best SSIM score: 0.9271302917943481
# Best FID score index: 392 Best FID score: 38.97646855433744
    