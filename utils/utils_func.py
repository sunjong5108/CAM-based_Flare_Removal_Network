import torch
import numpy as np
import os 
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
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
    w, h = ori_image.size

    # PIL image -> Tensor
    totensor = transforms.ToTensor()
    ori_image_tensor = totensor(ori_image)
    flare_image_tensor = totensor(flare_image)
    _, h, w = ori_image_tensor.shape
    flare_image_tensor = TF.resize(flare_image_tensor, size=[h, w])

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
    path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'path_gamma_df.csv'))

def flare_synthesized_img_split(flare_synthetic_path, flare_synthetic_noise_save_path, train_path, test_path, gamma_path, train_save_dir, test_save_dir):
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
    train_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'train_path_gamma_df.csv'))
    test_path_gamma_df = pd.DataFrame.from_dict(test_path_gamma_dict, orient='index')
    test_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'test_path_gamma_df.csv'))

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
                                gamma_train_path, gamma_test_path, flare_synthetic_noise_save_path):
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
    fold1_train_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'fold1_train_path_gamma_df.csv'))
    fold2_train_path_gamma_df = pd.DataFrame.from_dict(fold2_train_path_gamma_dict, orient='index')
    fold2_train_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'fold2_train_path_gamma_df.csv'))
    fold1_val_path_gamma_df = pd.DataFrame.from_dict(fold1_val_path_gamma_dict, orient='index')
    fold1_val_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'fold1_val_path_gamma_df.csv'))
    fold2_val_path_gamma_df = pd.DataFrame.from_dict(fold2_val_path_gamma_dict, orient='index')
    fold2_val_path_gamma_df.to_csv(os.path.join(os.path.join('/workspace/Datas/', 'KITTI_flare_gamma'), 'fold2_val_path_gamma_df.csv'))

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

def shuffle_data(is_label, file_input_list, file_label_list):
    '''
    Data shuffle for making 2-fold datasets
    file_input_list: input path
    file_label_list: label path
    '''
    idx_list = np.arange(0, len(file_input_list))
    
    np.random.shuffle(idx_list)
    if is_label == True:
        result_input_list = [file_input_list[idx] for idx in idx_list]
        result_label_list = [file_label_list[idx] for idx in idx_list]
        return result_input_list, result_label_list
    else:
        result_input_list = [file_input_list[idx] for idx in idx_list]
        return result_input_list

def two_fold_dataset_generation(is_label, total_input_path, total_label_path, save_path_list):
    '''
    1. shuffle data in total input path and total label path
    2. Divide half path (shuffle data path)
    3. save image each data path

    total_input_path: total_input_path
    total_label_path: total_label_path
    train_input_save_path: save_path_list[-4]
    train_label_save_path: save_path_list[-3]
    test_input_save_path: save_path_list[-2]
    test_label_save_path: save_path_list[-1]
    '''
    shuffle_input_path, shuffle_label_path = shuffle_data(is_label, total_input_path, total_label_path)
    total_num = len(total_input_path)
    fold_num = total_num // 2

    fold_1_test_input_data = shuffle_input_path[:fold_num]
    fold_1_test_label_data = shuffle_label_path[:fold_num]
    
    fold_1_train_input_data = shuffle_input_path[fold_num:]
    fold_1_train_label_data = shuffle_label_path[fold_num:]

    for path in save_path_list:
        os.makedirs(path, exist_ok=True)
    
    if is_label == True:
        for i, (input, label) in enumerate(zip(fold_1_test_input_data, fold_1_test_label_data)):
            input_img = Image.open(input)
            label_img = Image.open(label)

            input_img.save(os.path.join(save_path_list[-2], fold_1_test_input_data[i].split('/')[-1]))
            label_img.save(os.path.join(save_path_list[-1], fold_1_test_label_data[i].split('/')[-1]))

        for i, (input, label) in enumerate(zip(fold_1_train_input_data, fold_1_train_label_data)):
            input_img = Image.open(input)
            label_img = Image.open(label)

            input_img.save(os.path.join(save_path_list[-4], fold_1_train_input_data[i].split('/')[-1]))
            label_img.save(os.path.join(save_path_list[-3], fold_1_train_label_data[i].split('/')[-1]))
    else:
        for i, input in enumerate(fold_1_test_input_data):
            input_img = Image.open(input)

            input_img.save(os.path.join(save_path_list[-1], fold_1_test_input_data[i].split('/')[-1]))

        for i, input in enumerate(fold_1_train_input_data):
            input_img = Image.open(input)

            input_img.save(os.path.join(save_path_list[-2], fold_1_train_input_data[i].split('/')[-1]))

def fold_train_val_split(is_label, train_input_path, train_label_path, save_path_list, valid_ratio):
    '''
    1. train data shuffle for validation set split
    2. validation set's length = training set'length * valid ratio
    3. save images

    train_input_path: train_input_path
    train_label_path: train_label_path
    test_input_path: test_input_path
    test_label_path: test_label_path
    train_input_save_path: save_path_list[-4]
    train_label_save_path: save_path_list[-3]
    valid_input_save_path: save_path_list[-2]
    valid_label_save_path: save_path_list[-1]
    '''
    shuffle_input_path, shuffle_label_path = shuffle_data(is_label, train_input_path, train_label_path)

    total_num = len(train_input_path)
    valid_num = int(total_num * valid_ratio)

    fold_val_input_data = shuffle_input_path[:valid_num]
    fold_val_label_data = shuffle_label_path[:valid_num]

    fold_train_input_data = shuffle_input_path[valid_num:]
    fold_train_label_data = shuffle_label_path[valid_num:]

    for path in save_path_list:
        os.makedirs(path, exist_ok=True)

    if is_label == True:
        for i, (input, label) in enumerate(zip(fold_val_input_data, fold_val_label_data)):
            input_img = Image.open(input)
            label_img = Image.open(label)

            input_img.save(os.path.join(save_path_list[-2], fold_val_input_data[i].split('/')[-1]))
            label_img.save(os.path.join(save_path_list[-1], fold_val_label_data[i].split('/')[-1]))

        for i, (input, label) in enumerate(zip(fold_train_input_data, fold_train_label_data)):
            input_img = Image.open(input)
            label_img = Image.open(label)

            input_img.save(os.path.join(save_path_list[-4], fold_train_input_data[i].split('/')[-1]))
            label_img.save(os.path.join(save_path_list[-3], fold_train_label_data[i].split('/')[-1]))
    else:
        for i, input in enumerate(fold_val_input_data):
            input_img = Image.open(input)

            input_img.save(os.path.join(save_path_list[-1], fold_val_input_data[i].split('/')[-1]))

        for i, input in enumerate(fold_train_input_data):
            input_img = Image.open(input)

            input_img.save(os.path.join(save_path_list[-2], fold_train_input_data[i].split('/')[-1]))

def calc_class_weights(data_loader, class_nums):
    z = np.zeros((class_nums,))
    for data in data_loader:
        sample = data[1].detach().numpy()
        labels = sample.astype(np.uint8)
        
        count = np.bincount(labels.reshape(-1), minlength=class_nums)
        z += count
        total_freq = np.sum(z)
        class_weights = []
        for freq in z:
            class_weight = 1 / (np.log(1.02 + (freq / total_freq)))
            class_weights.append(class_weight)

        ret = np.array(class_weights)

    return ret