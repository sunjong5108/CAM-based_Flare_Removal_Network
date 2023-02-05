import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
from models.flare_detect_cam import *
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from .SegAug import random_transforms_apply
from .SegLabel import rgb_to_mask
import numpy as np

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
                   
class TrainLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, flare_dir):
    super(TrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.flare_dir = flare_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]
    flare_path = self.flare_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    flare_img = Image.open(flare_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)
    flare_img = TF.to_tensor(flare_img)

    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
        flare_img = flare_img.flip(1)
    elif aug==2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
        flare_img = flare_img.flip(2)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, flare_img, filename

class TrainLoadDataset_v2(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(TrainLoadDataset_v2, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
    elif aug==2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class KittiTrainLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, flare_dir):
    super(KittiTrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.flare_dir = flare_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]
    flare_path = self.flare_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    flare_img = Image.open(flare_path)

    inp_img = TF.resize(inp_img, size=[400, 1200])
    tar_img = TF.resize(tar_img, size=[400, 1200])
    flare_img = TF.resize(flare_img, size=[400, 1200])

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)
    flare_img = TF.to_tensor(flare_img)

    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
        flare_img = flare_img.flip(1)
    elif aug==2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
        flare_img = flare_img.flip(2)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, flare_img, filename

class KittiTrainLoadDataset_v2(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(KittiTrainLoadDataset_v2, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    inp_img = TF.resize(inp_img, size=[400, 1200])
    tar_img = TF.resize(tar_img, size=[400, 1200])

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
    elif aug==2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class ValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, flare_dir):
    super(ValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.flare_dir = flare_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]
    flare_path = self.flare_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    flare_img = Image.open(flare_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)
    flare_img = TF.to_tensor(flare_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, flare_img, filename

class ValTestLoadDataset_v2(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(ValTestLoadDataset_v2, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class KittiValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, flare_dir):
    super(KittiValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.flare_dir = flare_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]
    flare_path = self.flare_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    flare_img = Image.open(flare_path)

    w, h = inp_img.size
    ori_size = (h, w)

    inp_img = TF.resize(inp_img, size=[400, 1200])
    tar_img = TF.resize(tar_img, size=[400, 1200])
    flare_img = TF.resize(flare_img, size=[400, 1200])

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)
    flare_img = TF.to_tensor(flare_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, flare_img, filename, ori_size

class KittiValTestLoadDataset_v2(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(KittiValTestLoadDataset_v2, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    w, h = inp_img.size
    ori_size = (h, w)

    inp_img = TF.resize(inp_img, size=[400, 1200])
    tar_img = TF.resize(tar_img, size=[400, 1200])

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename, ori_size


class SegTrainLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, patch_size):
    super(SegTrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.patch_size = patch_size
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length
    ps = self.patch_size

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    w,h = inp_img.size
    padw = ps-w if w<ps else 0
    padh = ps-h if h<ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw!=0 or padh!=0:
        inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
        tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

    ww, hh = inp_img.size

    rr     = random.randint(0, hh-ps)
    cc     = random.randint(0, ww-ps)

    aug    = random.randint(1, 7)

    # Data Augmentations
    inp_img, tar_img = random_transforms_apply(mode=aug, input=inp_img, label=tar_img)
    
    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    inp_img = TF.to_tensor(inp_img)
    tar_img = torch.from_numpy(np.array(tar_img, dtype=np.uint8))

    inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
    tar_img = tar_img[rr:rr+ps, cc:cc+ps]

    return inp_img, tar_img, filename

class SegValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(SegValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = torch.from_numpy(np.array(tar_img, dtype=np.uint8))

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class SegKittiTrainLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir, patch_size):
    super(SegKittiTrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.patch_size = patch_size
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length
    ps = self.patch_size

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path).convert('RGB')

    w,h = inp_img.size
    padw = ps-w if w<ps else 0
    padh = ps-h if h<ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw!=0 or padh!=0:
        inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
        tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

    ww, hh = inp_img.size

    rr     = random.randint(0, hh-ps)
    cc     = random.randint(0, ww-ps)

    aug    = random.randint(1, 7)

    # Data Augmentations
    inp_img, tar_img = random_transforms_apply(mode=aug, input=inp_img, label=tar_img)
    
    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    inp_img = TF.to_tensor(inp_img)
    tar_img = rgb_to_mask(np.array(tar_img, dtype=np.uint8), label_color_map)
    tar_img = np.argmax(tar_img, axis=0).astype(np.uint8)
    tar_img = torch.from_numpy(tar_img)

    inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
    tar_img = tar_img[rr:rr+ps, cc:cc+ps]

    return inp_img, tar_img, filename

class SegKittiValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(SegKittiValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path).convert('RGB')

    inp_img = TF.to_tensor(inp_img)
    tar_img = rgb_to_mask(np.array(tar_img, dtype=np.uint8), label_color_map)
    tar_img = np.argmax(tar_img, axis=0).astype(np.uint8)
    tar_img = torch.from_numpy(tar_img)
    
    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class DB3_ValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(DB3_ValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename