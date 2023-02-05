import torch
import torchvision
import torch.optim as optim
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from customdataset.get_load_dataset import *
from losses.losses import *
from models.proposed_model import *
from models.flare_detect_cam import *
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import pandas as pd
from utils.inception import *
from utils.metrics import *

def test(args):
    DB = args.database
    flare_test_dataset_path = args.test_input_path
    flare_test_label_path = args.test_label_path
    weights_save_path = args.weights_save_path
    best_model_pth = args.best_model_pth
    test_results_path = args.test_results_path
    FD_model_weights_path = args.fd_model_weights_path
    root = args.data_root
    save_root = args.save_root
    device = args.device
    eval_batch_size = args.eval_batch_size

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    incepv3 = InceptionV3([block_idx])
    incepv3 = incepv3.to(device)

    model_weights_save = os.path.join(save_root, weights_save_path)

    result_test_images_save = os.path.join(save_root, test_results_path)
    os.makedirs(result_test_images_save, exist_ok=True)       

    flare_fold_test_dataset_path = sorted([os.path.join(root, flare_test_dataset_path, file) 
                                        for file in os.listdir(os.path.join(root, flare_test_dataset_path))])

    flare_fold_test_label_path = sorted([os.path.join(root, flare_test_label_path, file) 
                                    for file in os.listdir(os.path.join(root, flare_test_label_path))])

    if DB == 'CamVid':
        test_datasets = get_val_test_dataset_v2(
        inp_dir=flare_fold_test_dataset_path,
        tar_dir=flare_fold_test_label_path
        )
    elif DB == 'KITTI':
        test_datasets = get_Kitti_val_test_dataset_v2(
        inp_dir=flare_fold_test_dataset_path,
        tar_dir=flare_fold_test_label_path,
        )

    test_loader = DataLoader(test_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    # MODEL
    vae = CAM_FRN(ch=3, blocks=5, dim=128, z_dim=512)

    LensFlareMask_extractor = CAM()
    checkpoint = torch.load(FD_model_weights_path)
    LensFlareMask_extractor.load_state_dict(checkpoint['model_state_dict'])

    for target_param_cam in LensFlareMask_extractor.parameters():
        target_param_cam.requires_grad = False

    vae.to(device)
    LensFlareMask_extractor.to(device)

    checkpoint = torch.load(os.path.join(model_weights_save, best_model_pth))
    vae.load_state_dict(checkpoint['vae_state_dict'])
    epoch = checkpoint['epoch']
    scale = [1.0, 0.5, 1.5, 2.0]

    Test_history = {'psnr': [], 'ssim': [], 'fid': []}
    # TEST
    vae.eval()
    with torch.no_grad():
        # METRIC
        def eval_step(engine, batch):
            return batch
        test_default_evaluator = Engine(eval_step)

        test_metric_ssim = SSIM(1.0)
        test_metric_ssim.attach(test_default_evaluator, 'ssim')

        test_metric_psnr = PSNR(1.0) 
        test_metric_psnr.attach(test_default_evaluator, 'psnr')

        epoch_start_time = time.time()
        epoch_test_psnr = 0
        epoch_test_ssim = 0
        epoch_test_fid = 0
        epoch_start_time = time.time()
        print('======> Test Start')
        for i, data in enumerate(tqdm(test_loader)):
            test_input = data[0].cuda()
            test_clean = data[1].cuda()
            file_name = data[-1]

            test_fmd_inputs_list = []
            size = (test_input.shape[2], test_clean.shape[3])
            for s in scale:
                target_size = (int(np.round(size[0]*s)), int(np.round(size[1]*s)))
                test_fmd_inputs_list.append(F.interpolate(test_input, size=target_size, mode='bicubic'))
            
            test_fmd_outputs = [LensFlareMask_extractor(img) for img in test_fmd_inputs_list]

            test_highres_cam = torch.sum(torch.stack([F.interpolate(o, size, mode='bilinear', align_corners=False) for o in test_fmd_outputs]), 0)
            
            test_highres_cam = test_highres_cam[:, 0, :, :].unsqueeze(1)
            test_highres_cam /= F.adaptive_avg_pool2d(test_highres_cam, (1, 1)) + 1e-5
            
            test_m = test_highres_cam.clone()
            test_m[test_m >= 0.2] = 1
            test_m[test_m < 0.2] = 0
            test_masked_input = test_input * (1-test_m) + test_m
            test_out, _, _, _ = vae(test_input, test_highres_cam, test_masked_input, test_m)

            test_fid = calculate_fretchet(test_out, test_clean, incepv3)
            test_psnr_state = test_default_evaluator.run([[test_out, test_clean]])
            test_ssim_state = test_default_evaluator.run([[test_out, test_clean]])

            epoch_test_psnr += test_psnr_state.metrics['psnr']
            epoch_test_ssim += test_ssim_state.metrics['ssim']
            epoch_test_fid += test_fid
        
            torchvision.utils.save_image(test_out, os.path.join(result_test_images_save, file_name[0] + '.png'))
            
    print('Epoch: {}\tTime: {:.4f}\tPSNR: {:.4f}\tSSIM: {:4f}\tFID: {:.4f}'.format(epoch, time.time() - epoch_start_time, epoch_test_psnr / len(test_loader), epoch_test_ssim / len(test_loader), epoch_test_fid / len(test_loader)))
    Test_history['psnr'].append(epoch_test_psnr / len(test_loader))
    Test_history['ssim'].append(epoch_test_ssim / len(test_loader))
    Test_history['fid'].append(epoch_test_fid / len(test_loader))
    Test_history = pd.DataFrame.from_dict(Test_history, orient='index')
    Test_history.to_csv(os.path.join(model_weights_save, 'test_history_v2.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test flare removal')
    parser.add_argument('--data_root', type=str, default='/workspace/Datas/')
    parser.add_argument('--database', type=str, default='CamVid')
    parser.add_argument('--test_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_test')
    parser.add_argument('--test_label_path', type=str, default='camvid_label_12_2fold_v2/fold1_test/inputs')
    parser.add_argument('--fd_model_weights_path', type=str, default='/workspace/DB1_proposed_model_fd_fold1_weights/model_168.pth')
    parser.add_argument('--weights_save_path', type=str, default='DB1_proposed_cam_frn_fold1_weights-2_fold')
    parser.add_argument('--best_model_pth', type=str)
    parser.add_argument('--test_results_path', type=str, default='DB1_proposed_cam_frn_fold1_test_results-2_fold')
    parser.add_argument('--save_root', type=str, default='/workspace/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    args = parser.parse_args()

    test(args)