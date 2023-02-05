import argparse
import os
import numpy as np
import cv2
from customdataset.get_load_dataset import *
from customdataset.SegLabel import *
from utils.utils_func import *
from utils.metrics import *

def pixel_acc(pred, true, class_num):
    smooth = 1e-6

    pred_1d = pred.reshape(-1)
    true_1d = true.reshape(-1)
    
    categorical_array = (class_num * true_1d) + pred_1d
    
    confusion_mat_1d = np.bincount(categorical_array, minlength=class_num*class_num)
    confusion_mat = confusion_mat_1d.reshape((class_num, class_num))
    confusion_mat[-1] = 0
    correct = np.sum(np.diag(confusion_mat))
    total = np.sum(confusion_mat)
    pixel_acc = correct / (total + smooth)

    return pixel_acc

def class_acc(pred, true, class_num):
    smooth = 1e-6

    pred_1d = pred.reshape(-1)
    true_1d = true.reshape(-1)
    
    categorical_array = (class_num * true_1d) + pred_1d
    
    confusion_mat_1d = np.bincount(categorical_array, minlength=class_num*class_num)
    confusion_mat = confusion_mat_1d.reshape((class_num, class_num))
    confusion_mat[-1] = 0
    
    correct_per_class = np.diag(confusion_mat)
    total_per_class = np.sum(confusion_mat, axis=1)
    per_class_acc = correct_per_class / (total_per_class + smooth)
    avg_per_class_acc = np.nanmean(per_class_acc)
    return avg_per_class_acc

def calc_pixel_and_class_acc(pred_seg_path, label_path, label_color_map):
  save_pixel_acc = []
  save_class_acc = []
  cnt = 0

  for step, (input, label) in enumerate(zip(pred_seg_path, label_path)):
    cnt += 1
    test_pred = cv2.imread(input)
    test_pred = cv2.cvtColor(test_pred, cv2.COLOR_BGR2RGB)

    test_pred = rgb_to_mask(test_pred, label_color_map)
    test_pred = np.transpose(test_pred, (1, 2, 0))
    test_pred = np.argmax(test_pred, axis=2)

    test_label = cv2.imread(label)
    test_label = cv2.cvtColor(test_label, cv2.COLOR_BGR2RGB)
    test_label = rgb_to_mask(mask_to_rgb(test_label[:, :, 0], pred=False, color_map=label_color_map), label_color_map)
    test_label = np.transpose(test_label, (1, 2, 0))
    test_label = np.argmax(test_label, axis=2)

    save_pixel_acc.append(pixel_acc(test_pred, test_label, 12))
    save_class_acc.append(class_acc(test_pred, test_label, 12))

  print(f'test pixel acc: {np.mean(save_pixel_acc)}, test class acc: {np.mean(save_class_acc)}')
  return np.mean(save_pixel_acc), np.mean(save_class_acc)

def clac_class_acc_pixel_acc(args):
    root = args.data_root
    pred_path = args.pred_path
    label_path = args.label_path

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

    pred_dataset_path = sorted([os.path.join(root, pred_path, file) 
                                        for file in os.listdir(os.path.join(root, pred_path))])

    test_label_path = sorted([os.path.join(root, label_path, file) 
                                    for file in os.listdir(os.path.join(root, label_path))])

    _ = calc_pixel_and_class_acc(pred_dataset_path, test_label_path, label_color_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate class acc and pixel acc')
    parser.add_argument('--data_root', type=str, default='/workspace/Datas/')
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--label_path', type=str, default='camvid_label_12_2fold_v2_5_fold/fold1_label')
    args = parser.parse_args()

    clac_class_acc_pixel_acc(args)
