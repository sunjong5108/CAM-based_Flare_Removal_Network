import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms.functional as TF
import os

class FDTrainLoadDataset(Dataset):
  def __init__(self, inp_dir):
    super(FDTrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]

    inp_img = Image.open(inp_path)
    inp_img = TF.to_tensor(inp_img)
    inp_img = TF.normalize(inp_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if inp_path.split('/')[3].split('_')[0] == 'flare':
        target = torch.Tensor([1, 0]).float()
    elif inp_path.split('/')[3].split(' ')[0].split('_')[0] == 'camvid': 
        target = torch.Tensor([0, 1]).float()
    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        if inp_path.split('/')[3].split('_')[0] == 'flare':
            target = torch.Tensor([1, 0]).float()
        elif inp_path.split('/')[3].split(' ')[0].split('_')[0] == 'camvid': 
            target = torch.Tensor([0, 1]).float()
    elif aug==2:
        inp_img = inp_img.flip(2)
        if inp_path.split('/')[3].split('_')[0] == 'flare':
            target = torch.Tensor([1, 0]).float()
        elif inp_path.split('/')[3].split(' ')[0].split('_')[0] == 'camvid': 
            target = torch.Tensor([0, 1]).float()

    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] 

    return inp_img, target, filename

class FDValTestLoadDataset(Dataset):
  def __init__(self, inp_dir):
    super(FDValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]

    inp_img = Image.open(inp_path)
    inp_img = TF.to_tensor(inp_img)
    inp_img = TF.normalize(inp_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if inp_path.split('/')[3].split('_')[0] == 'flare':
        target = torch.Tensor([1, 0]).float()
    elif inp_path.split('/')[3].split(' ')[0].split('_')[0] == 'camvid': 
        target = torch.Tensor([0, 1]).float()

    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] 

    return inp_img, target, filename

class Kitti_FDTrainLoadDataset(Dataset):
  def __init__(self, inp_dir):
    super(Kitti_FDTrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]

    inp_img = Image.open(inp_path)
    inp_img = TF.resize(inp_img, size=[400, 1200])
    inp_img = TF.to_tensor(inp_img)
    inp_img = TF.normalize(inp_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if inp_path.split('/')[3].split('_')[0] == 'flare':
        target = torch.Tensor([1, 0]).float()
    elif inp_path.split('/')[3].split('_')[0].split('_')[0] == 'KITTI': 
        target = torch.Tensor([0, 1]).float()
    aug    = random.randint(0, 3)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(1)
        if inp_path.split('/')[3].split('_')[0] == 'flare':
            target = torch.Tensor([1, 0]).float()
        elif inp_path.split('/')[3].split('_')[0].split('_')[0] == 'KITTI': 
            target = torch.Tensor([0, 1]).float()
    elif aug==2:
        inp_img = inp_img.flip(2)
        if inp_path.split('/')[3].split('_')[0] == 'flare':
            target = torch.Tensor([1, 0]).float()
        elif inp_path.split('/')[3].split('_')[0].split('_')[0] == 'KITTI': 
            target = torch.Tensor([0, 1]).float()

    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] 

    return inp_img, target, filename

class Kitti_FDValTestLoadDataset(Dataset):
  def __init__(self, inp_dir):
    super(Kitti_FDValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]

    inp_img = Image.open(inp_path)
    inp_img = TF.resize(inp_img, size=[400, 1200])
    inp_img = TF.to_tensor(inp_img)
    inp_img = TF.normalize(inp_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if inp_path.split('/')[3].split('_')[0] == 'flare':
        target = torch.Tensor([1, 0]).float()
    elif inp_path.split('/')[3].split('_')[0].split('_')[0] == 'KITTI': 
        target = torch.Tensor([0, 1]).float()

    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] 

    return inp_img, target, filename