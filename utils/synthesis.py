import torch
import torch.nn as nn
import numpy as np
import time
import os 
import cv2
from tqdm import tqdm
import math
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from PIL import Image
import warnings

warnings.simplefilter('ignore', Warning, lineno=0)

def remove_background(image):
    eps = 1e-7
    image_min = image.amin([-1, -2], keepdim=True)
    image_max = image.amax([-1, -2], keepdim=True)
    return (image - image_min) * image_max / (image_max - image_min + eps)

def normalize_white_balance(image):
    eps = 1e-7
    channel_mean = image.mean([-1, -2], keepdim=True)
    max_of_mean = channel_mean.amax([-1, -2, -3], keepdim=True)
    normalized = max_of_mean * image / (channel_mean + eps)

    return normalized

def quantize_8(image):
    return (image * 255).to(torch.uint8).float() * (1.0/255.0)

def flare_synthesize(ori_image_path, flare_path, noise, flare_max_gain):
    # 1. adjust gamma -> linearization
    # 2. original_img_linearization = original_img_linearization + noise
    # 3. original_img_linearization = random gain * original_img_linearization
    # 4. original_img_linearization + flare_linearization
    random_gamma = np.random.uniform(1.8, 2.2)
    random_gain = np.random.uniform(0.5, 1.0)
    random_rgb_gain = np.random.uniform(1, flare_max_gain, size=[3])
    random_offset = np.random.uniform(-0.05, 0.05)

    ori_image = Image.open(ori_image_path)
    flare_image = Image.open(flare_path)

    # PIL image -> Tensor
    totensor = transforms.ToTensor()
    ori_image_tensor = totensor(ori_image)
    flare_image_tensor = totensor(flare_image)

    ori_image_linear = transforms.functional.adjust_gamma(ori_image_tensor, gamma=random_gamma)
    flare_image_linear = transforms.functional.adjust_gamma(flare_image_tensor, gamma=random_gamma)

    # clean image + noise, gain * clean image

    sigma = np.abs(np.random.normal(0, noise))
    chi_noise = np.random.normal(0, sigma, (ori_image_linear.shape))

    ori_image_linear += chi_noise

    ori_image_linear = (random_gain * ori_image_linear).clamp_(0.0, 1.0)

    # flare images processing
    flare_image_linear = remove_background(flare_image_linear)
    flare_image_linear = flare_image_linear.clamp_(0.0, 1.0)

    random_rgb_gain = random_rgb_gain.reshape([3, 1, 1])
    flare_image_linear *= random_rgb_gain
    flare_image_linear = flare_image_linear.clamp_(0.0, 1.0)

    flare_image_linear = transforms.functional.gaussian_blur(flare_image_linear, kernel_size=[3, 3], sigma=(0.1, 3))

    flare_image_linear = flare_image_linear + random_offset
    flare_image_linear = flare_image_linear.clamp_(0.0, 1.0)

    # synthesize image
    synthesized_img_linear = ori_image_linear + flare_image_linear
    synthesized_img_linear = synthesized_img_linear.clamp_(0.0, 1.0)

    synthesized_img = transforms.functional.adjust_gamma(synthesized_img_linear, gamma=1/random_gamma)
    synthesized_img = synthesized_img.clamp_(0.0, 1.0)

    synthesized_img = quantize_8(synthesized_img)

    return synthesized_img, random_gamma

def save_synthesized_image(Camvid_ori_img_dataset_path, Camvid_flare_img_dataset_path, flare_synthetic_noise_save_path):
    topilimage = transforms.ToPILImage()
    path_gamma_dict = {'filename': []}

    for i, (ori_img_path, flare_img_path) in enumerate(tqdm(zip(Camvid_ori_img_dataset_path, Camvid_flare_img_dataset_path))):
        result_img, gamma = flare_synthesize(ori_img_path, flare_img_path, noise=0.01, flare_max_gain=1.1)

        path_gamma_dict['filename'].append(ori_img_path.split('/')[-1] + '-' + str(gamma))
        
        result_img_PIL = topilimage(result_img)

        result_img_PIL.save(os.path.join(flare_synthetic_noise_save_path, ori_img_path.split('/')[-1]))

    path_gamma_df = pd.DataFrame.from_dict(path_gamma_dict, orient='index')
    path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'path_gamma_df.csv'))

def flare_synthesized_img_split(flare_synthetic_path, train_path, test_path, gamma_path, train_save_dir, test_save_dir):
    train_path_gamma_dict = {'filename': []}
    test_path_gamma_dict = {'filename': []}

    for j, img_path in enumerate(tqdm(flare_synthetic_path)):
        for i, input in enumerate(train_path):
            if img_path.split('/')[-1] == input.split('/')[-1]:
                input_img = Image.open(img_path)
                input_img.save(os.path.join(train_save_dir, img_path.split('/')[-1]))
            if gamma_path[j].split('-')[0] == input.split('/')[-1]:
                train_path_gamma_dict['filename'].append(gamma_path[j])

        for i, input in enumerate(test_path):
            if img_path.split('/')[-1] == input.split('/')[-1]:
                input_img = Image.open(img_path)
                input_img.save(os.path.join(test_save_dir, img_path.split('/')[-1]))
            if gamma_path[j].split('-')[0] == input.split('/')[-1]:
                test_path_gamma_dict['filename'].append(gamma_path[j])

    train_path_gamma_df = pd.DataFrame.from_dict(train_path_gamma_dict, orient='index')
    train_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'train_path_gamma_df.csv'))
    test_path_gamma_df = pd.DataFrame.from_dict(test_path_gamma_dict, orient='index')
    test_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'test_path_gamma_df.csv'))

def flare_img_split(flare_path, train_path, test_path, train_save_dir, test_save_dir):
    for j, img_path in enumerate(tqdm(flare_path)):
        for i, input in enumerate(train_path):
            if img_path.split('/')[-1] == input.split('/')[-1]:
                input_img = Image.open(img_path)
                input_img.save(os.path.join(train_save_dir, img_path.split('/')[-1]))

        for i, input in enumerate(test_path):
            if img_path.split('/')[-1] == input.split('/')[-1]:
                input_img = Image.open(img_path)
                input_img.save(os.path.join(test_save_dir, img_path.split('/')[-1]))

def flare_synthesized_fold_split(flare_include_train_path, fold1_train_path, fold1_val_path, fold1_train_save_dir, fold1_val_save_dir, 
                                flare_include_test_path, fold2_train_path, fold2_val_path, fold2_train_save_dir, fold2_val_save_dir,
                                gamma_train_path, gamma_test_path):
    fold1_train_path_gamma_dict = {'filename': []}
    fold2_train_path_gamma_dict = {'filename': []}
    fold1_val_path_gamma_dict = {'filename': []}
    fold2_val_path_gamma_dict = {'filename': []}
    for i, path1 in enumerate(tqdm(flare_include_train_path)):
        for j, path2 in enumerate(fold1_train_path):
            if path1.split('/')[-1] == path2.split('/')[-1]:
                input_img1 = Image.open(path1)
                input_img1.save(os.path.join(fold1_train_save_dir, path1.split('/')[-1]))
            if gamma_train_path[i].split('-')[0] == path2.split('/')[-1]:
                fold1_train_path_gamma_dict['filename'].append(gamma_train_path[i])

        for k, path3 in enumerate(fold1_val_path):
            if path1.split('/')[-1] == path3.split('/')[-1]:
                input_img2 = Image.open(path1)
                input_img2.save(os.path.join(fold1_val_save_dir, path1.split('/')[-1]))
            if gamma_train_path[i].split('-')[0] == path3.split('/')[-1]:
                fold1_val_path_gamma_dict['filename'].append(gamma_train_path[i])


    for i, path1 in enumerate(tqdm(flare_include_test_path)):
        for j, path2 in enumerate(fold2_train_path):
            if path1.split('/')[-1] == path2.split('/')[-1]:
                input_img3 = Image.open(path1)
                input_img3.save(os.path.join(fold2_train_save_dir, path1.split('/')[-1]))
            if gamma_test_path[i].split('-')[0] == path2.split('/')[-1]:
                fold2_train_path_gamma_dict['filename'].append(gamma_test_path[i])

        for k, path3 in enumerate(fold2_val_path):
            if path1.split('/')[-1] == path3.split('/')[-1]:
                input_img4 = Image.open(path1)
                input_img4.save(os.path.join(fold2_val_save_dir, path1.split('/')[-1]))
            if gamma_test_path[i].split('-')[0] == path3.split('/')[-1]:
                fold2_val_path_gamma_dict['filename'].append(gamma_test_path[i])

    fold1_train_path_gamma_df = pd.DataFrame.from_dict(fold1_train_path_gamma_dict, orient='index')
    fold1_train_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'fold1_train_path_gamma_df.csv'))
    fold2_train_path_gamma_df = pd.DataFrame.from_dict(fold2_train_path_gamma_dict, orient='index')
    fold2_train_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'fold2_train_path_gamma_df.csv'))
    fold1_val_path_gamma_df = pd.DataFrame.from_dict(fold1_val_path_gamma_dict, orient='index')
    fold1_val_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'fold1_val_path_gamma_df.csv'))
    fold2_val_path_gamma_df = pd.DataFrame.from_dict(fold2_val_path_gamma_dict, orient='index')
    fold2_val_path_gamma_df.to_csv(os.path.join('/workspace/Flare_removal/', 'fold2_val_path_gamma_df.csv'))

def flare_fold_split(flare_train_path, fold1_train_path, fold1_val_path, fold1_train_save_dir, fold1_val_save_dir, 
                    flare_test_path, fold2_train_path, fold2_val_path, fold2_train_save_dir, fold2_val_save_dir):
    for i, path1 in enumerate(tqdm(flare_train_path)):
        for j, path2 in enumerate(fold1_train_path):
            if path1.split('/')[-1] == path2.split('/')[-1]:
                input_img1 = Image.open(path1)
                input_img1.save(os.path.join(fold1_train_save_dir, path1.split('/')[-1]))
        for k, path3 in enumerate(fold1_val_path):
            if path1.split('/')[-1] == path3.split('/')[-1]:
                input_img2 = Image.open(path1)
                input_img2.save(os.path.join(fold1_val_save_dir, path1.split('/')[-1]))


    for i, path1 in enumerate(tqdm(flare_test_path)):
        for j, path2 in enumerate(fold2_train_path):
            if path1.split('/')[-1] == path2.split('/')[-1]:
                input_img3 = Image.open(path1)
                input_img3.save(os.path.join(fold2_train_save_dir, path1.split('/')[-1]))

        for k, path3 in enumerate(fold2_val_path):
            if path1.split('/')[-1] == path3.split('/')[-1]:
                input_img4 = Image.open(path1)
                input_img4.save(os.path.join(fold2_val_save_dir, path1.split('/')[-1]))

if __name__ == '__main__':
    root = '/workspace/datas/'
    Camvid_fold_train_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs', file) 
                             for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs'))])

    Camvid_fold_train_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'labels', file) 
                            for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'labels'))])

    Camvid_fold_test_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs'))])

    Camvid_fold_test_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'labels'))])

    Camvid_fold1_train_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'inputs'))])

    Camvid_fold1_train_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'labels', file) 
                            for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_train', 'labels'))])

    Camvid_fold1_val_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'inputs'))])

    Camvid_fold1_val_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold1_val', 'labels'))])               

    Camvid_fold1_test_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'inputs'))])

    Camvid_fold1_test_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'test', 'labels'))])

    Camvid_fold2_train_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'inputs'))])

    Camvid_fold2_train_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'labels', file) 
                            for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_train', 'labels'))])

    Camvid_fold2_val_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'inputs'))])

    Camvid_fold2_val_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'fold2_val', 'labels'))])               

    Camvid_fold2_test_dataset_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'inputs'))])

    Camvid_fold2_test_label_path = sorted([os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'labels', file) 
                                for file in os.listdir(os.path.join(root, 'camvid_label_12_2fold_v2', 'train', 'labels'))])

    Camvid_ori_img_dataset_path = sorted([os.path.join(root, 'camvid_label_12_original', file) 
                                      for file in os.listdir(os.path.join(root, 'camvid_label_12_original'))])

    Camvid_flare_img_dataset_path = sorted([os.path.join(root, 'gamma2_2_flare_v2', file) 
                                        for file in os.listdir(os.path.join(root, 'gamma2_2_flare_v2'))])

    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4'), exist_ok=True)

    flare_synthetic_noise_save_path = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4')

    print('save synthesized image')
    save_synthesized_image(Camvid_ori_img_dataset_path, Camvid_flare_img_dataset_path, flare_synthetic_noise_save_path)

    flare_synthetic_noise_path = sorted([os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', file) 
                                for file in os.listdir(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4'))])
    gamma_df = pd.read_csv('/workspace/Flare_removal/path_gamma_df.csv', index_col=0)
    gamma_path = gamma_df.values.tolist()[0]

    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'test'), exist_ok=True)
    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'test'), exist_ok=True)

    flare_synthetic_train_save_path = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'train')
    flare_synthetic_test_save_path = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'test')
    flare_train_save_path = os.path.join(root, 'gamma2_2_flare_v2', 'train')
    flare_test_save_path = os.path.join(root, 'gamma2_2_flare_v2', 'test')

    print('train test split')
    flare_synthesized_img_split(flare_synthetic_noise_path, Camvid_fold_train_dataset_path, Camvid_fold_test_dataset_path, gamma_path, flare_synthetic_train_save_path, flare_synthetic_test_save_path)

    flare_img_split(Camvid_flare_img_dataset_path, Camvid_fold_train_dataset_path, Camvid_fold_test_dataset_path, flare_train_save_path, flare_test_save_path)

    Camvid_with_flare_train_dataset_path = sorted([os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'train', file) 
                                for file in os.listdir(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'train'))])

    Camvid_with_flare_test_dataset_path = sorted([os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'test', file) 
                                for file in os.listdir(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'test'))])

    flare_train_dataset_path = sorted([os.path.join(root, 'gamma2_2_flare_v2', 'train', file) 
                                for file in os.listdir(os.path.join(root, 'gamma2_2_flare_v2', 'train'))])

    flare_test_dataset_path = sorted([os.path.join(root, 'gamma2_2_flare_v2', 'test', file) 
                                for file in os.listdir(os.path.join(root, 'gamma2_2_flare_v2', 'test'))])

    train_gamma_df = pd.read_csv('/workspace/Flare_removal/train_path_gamma_df.csv', index_col=0)
    train_gamma_path = train_gamma_df.values.tolist()[0]
    test_gamma_df = pd.read_csv('/workspace/Flare_removal/test_path_gamma_df.csv', index_col=0)
    test_gamma_path = test_gamma_df.values.tolist()[0]

    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold1_train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold1_val'), exist_ok=True)
    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold2_train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold2_val'), exist_ok=True)

    fold1_train_save_dir = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold1_train')
    fold1_val_save_dir = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold1_val')
    fold2_train_save_dir = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold2_train')
    fold2_val_save_dir = os.path.join(root, 'flare synthesized CamVid 12label Dataset_0518_4', 'fold2_val')

    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'fold1_train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'fold1_val'), exist_ok=True)
    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'fold2_train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'gamma2_2_flare_v2', 'fold2_val'), exist_ok=True)

    fold1_flare_train_save_dir = os.path.join(root, 'gamma2_2_flare_v2', 'fold1_train')
    fold1_flare_val_save_dir = os.path.join(root, 'gamma2_2_flare_v2', 'fold1_val')
    fold2_flare_train_save_dir = os.path.join(root, 'gamma2_2_flare_v2', 'fold2_train')
    fold2_flare_val_save_dir = os.path.join(root, 'gamma2_2_flare_v2', 'fold2_val')

    print('fold split')
    flare_synthesized_fold_split(Camvid_with_flare_train_dataset_path, Camvid_fold1_train_dataset_path, Camvid_fold1_val_dataset_path, fold1_train_save_dir, fold1_val_save_dir,
                                Camvid_with_flare_test_dataset_path, Camvid_fold2_train_dataset_path, Camvid_fold2_val_dataset_path, fold2_train_save_dir, fold2_val_save_dir,
                                train_gamma_path, test_gamma_path)

    flare_fold_split(flare_train_dataset_path, Camvid_fold1_train_dataset_path, Camvid_fold1_val_dataset_path, fold1_flare_train_save_dir, fold1_flare_val_save_dir,
                    flare_test_dataset_path, Camvid_fold2_train_dataset_path, Camvid_fold2_val_dataset_path, fold2_flare_train_save_dir, fold2_flare_val_save_dir)