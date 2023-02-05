import argparse
import os
import torch
import numpy as np
from PIL import Image
from customdataset.get_load_dataset import *
from customdataset.SegLabel import *
from utils.utils_func import *
from segmodel.DeepLabV3_plus import DeepLab
from utils.metrics import *

def predict_output_(model, save_root, fold, test_loader, device, label_color_map):
    class_nums = len(label_color_map)
    class_names = list(label_color_map.keys())
    model.eval()
    with torch.no_grad():
        test_total_batch = len(test_loader)
        test_total_class_ious = np.zeros((test_total_batch * 1, class_nums))

        print("======> Test Start")
        for i, data in enumerate(tqdm(test_loader), 0):
            test_batch_ious_class = np.zeros((1, class_nums))
            test_input = data[0].to(device)
            test_label = data[1].to(device)
            filename = data[2]
            test_final_out = model(test_input)
            test_final_out_ = np.argmax(test_final_out.detach().cpu().numpy(), axis=1)

            for j in range(1):
                test_batch_ious_class[j] = iou_calc(torch.argmax(test_final_out.to('cpu'), dim=1)[j], test_label[j].to('cpu'), void=True, class_num=class_nums)

            test_total_class_ious[i*1:(i+1)*1, :] = test_batch_ious_class[:, :]

            output = np.zeros((test_final_out.shape[0], test_final_out.shape[2], test_final_out.shape[3], 3), dtype=np.uint8)

            for k, name in enumerate(class_names):
                output[test_final_out_ == k] = label_color_map[name]
            
            output = np.squeeze(output, axis=0)
            output_pil = Image.fromarray(output)
            segmentation_save = os.path.join(save_root, 'DB1_' + fold + '_segmentation_map')
            os.makedirs(segmentation_save, exist_ok=True)

            output_pil = output_pil.save(os.path.join(segmentation_save, filename[0] + '.png'))

    test_class_ious_per_epoch = np.nanmean(test_total_class_ious, axis=0)
    test_epoch_miou = np.nanmean(test_class_ious_per_epoch, axis=0)
    print(f'test miou: {test_class_ious_per_epoch} {test_epoch_miou}')

def test(args):
    DB = args.database
    root = args.data_root
    test_input_path = args.test_input_path
    test_label_path = args.test_label_path
    model_weight_path = args.model_weight_path
    fold = args.fold
    save_root = args.save_root
    device = args.device
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
    
    fold_input_path = sorted([os.path.join(root, test_input_path, file) 
                                for file in os.listdir(os.path.join(root, test_input_path))])
    fold_label_path = sorted([os.path.join(root, test_label_path, file) 
                                for file in os.listdir(os.path.join(root, test_label_path))])

    if DB == 'CamVid':
        test_datasets_fold = SEG_get_val_test_dataset(
            inp_dir=fold_input_path,
            tar_dir=fold_label_path
        )
    elif DB == 'KITTI':
        test_datasets_fold = SEG_Kitti_get_val_test_dataset(
            inp_dir=fold_input_path,
            tar_dir=fold_label_path
        )
    
    test_loader_fold = DataLoader(test_datasets_fold, batch_size=1, shuffle=False, drop_last=False)

    deeplabv3_plus_fold_load = DeepLab(output_stride=8, num_classes=12)
    fold1_check_point = torch.load(model_weight_path)
    deeplabv3_plus_fold_load.load_state_dict(fold1_check_point['model_state_dict'])

    predict_output_(deeplabv3_plus_fold_load.to(device), save_root, fold, test_loader_fold, device, label_color_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test deeplabV3_plus')
    parser.add_argument('--data_root', type=str, default='/workspace/Datas/')
    parser.add_argument('--database', type=str, default='CamVid')
    parser.add_argument('--fold', type=str, default='fold1')
    parser.add_argument('--test_input_path', type=str, default='camvid_label_12_2fold_v2_5_fold/fold1_input')
    parser.add_argument('--test_label_path', type=str, default='camvid_label_12_2fold_v2_5_fold/fold1_label')
    parser.add_argument('--model_weight_path', type=str, default='/workspace/DB1_proposed_segmodel_fold1_weights-5_fold/model_best_miou.pth')
    parser.add_argument('--save_root', type=str, default='/workspace/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    test(args)
