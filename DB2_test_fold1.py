import torch
import torchvision
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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

# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
incepv3 = InceptionV3([block_idx])
incepv3 = incepv3.cuda()

save_root = '/workspace/'
model_weights_save = os.path.join(save_root, 'DB2_proposed_model_fold1_weights_with_CAM_tv')

result_test_images_save = os.path.join(save_root, 'DB2_proposed_model_fold1_test_images_with_CAM_tv')
os.makedirs(result_test_images_save, exist_ok=True)

root = '/workspace/Datas/'         

Kitti_flare_fold1_test_dataset_path = sorted([os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'test', file) 
                                for file in os.listdir(os.path.join(root, 'flare_synthesized_KITTI_Dataset_1013', 'test'))])

Kitti_flare_fold1_test_label_path = sorted([os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'KITTI_ori_2_fold', 'test', 'inputs'))])

Kitti_flare_fold1_test_flare_path = sorted([os.path.join(root, 'KITTI_flare_v2', 'test', file) 
                            for file in os.listdir(os.path.join(root, 'KITTI_flare_v2', 'test'))])
    

NUM_EPOCH = 400
lr = 1e-4

test_datasets = get_Kitti_val_test_dataset(
    inp_dir=Kitti_flare_fold1_test_dataset_path,
    tar_dir=Kitti_flare_fold1_test_label_path,
    flare_dir=Kitti_flare_fold1_test_flare_path
)
test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, drop_last=False)

# MODEL
cam_frn = CAM_FRN(ch=3, blocks=5, dim=128, z_dim=512)
discriminator = Discriminator(3, use_sigmoid=True)

LensFlareMask_extractor = CAM()
checkpoint = torch.load(os.path.join('/workspace/DB2_proposed_model_fd_fold1_weights', 'model_195.pth'))
LensFlareMask_extractor.load_state_dict(checkpoint['model_state_dict'])

for target_param_cam in LensFlareMask_extractor.parameters():
    target_param_cam.requires_grad = False

cam_frn.cuda()
discriminator.cuda()
LensFlareMask_extractor.cuda()

lr=1e-4
gen_optimizer = optim.Adam(cam_frn.parameters(), lr=lr)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr*0.1)

checkpoint = torch.load(os.path.join(model_weights_save, 'model_343.pth'))
cam_frn.load_state_dict(checkpoint['vae_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
epoch = checkpoint['epoch']
scale = [1.0, 0.5, 1.5, 2.0]
strided_up_size = (360, 480)

Test_history = {'psnr': [], 'ssim': [], 'fid': []}
# TEST
cam_frn.eval()
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
        file_name = data[3]
        ori_size = data[-1]

        test_fmd_inputs_list = []
        size = (test_clean.shape[2], test_clean.shape[3])
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
        test_out, _, _, _ = cam_frn(test_input, test_highres_cam, test_masked_input, test_m)

        test_out = TF.resize(test_out, size=ori_size, interpolation=TF.InterpolationMode.BICUBIC)
        test_clean = TF.resize(test_clean, size=ori_size, interpolation=TF.InterpolationMode.BICUBIC)
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
Test_history.to_csv(os.path.join(model_weights_save, 'test_history.csv'))
