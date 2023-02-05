from .customdateset import *
from .flare_detect_dataset import *

def get_train_dataset(inp_dir, tar_dir, flare_dir):
  return TrainLoadDataset(inp_dir, tar_dir, flare_dir)

def get_train_dataset_v2(inp_dir, tar_dir):
  return TrainLoadDataset_v2(inp_dir, tar_dir)

def get_Kitti_train_dataset(inp_dir, tar_dir, flare_dir):
  return KittiTrainLoadDataset(inp_dir, tar_dir, flare_dir)

def get_Kitti_train_dataset_v2(inp_dir, tar_dir):
  return KittiTrainLoadDataset_v2(inp_dir, tar_dir)

def get_DB3_val_test_dataset(inp_dir, tar_dir):
  return DB3_ValTestLoadDataset(inp_dir, tar_dir)

def get_val_test_dataset(inp_dir, tar_dir, flare_dir):
  return ValTestLoadDataset(inp_dir, tar_dir, flare_dir)

def get_val_test_dataset_v2(inp_dir, tar_dir):
  return ValTestLoadDataset_v2(inp_dir, tar_dir)

def get_Kitti_val_test_dataset(inp_dir, tar_dir, flare_dir):
  return KittiValTestLoadDataset(inp_dir, tar_dir, flare_dir)

def get_Kitti_val_test_dataset_v2(inp_dir, tar_dir):
  return KittiValTestLoadDataset_v2(inp_dir, tar_dir)

def FD_get_train_dataset(inp_dir):
  return FDTrainLoadDataset(inp_dir)

def FD_get_val_test_dataset(inp_dir):
  return FDValTestLoadDataset(inp_dir)

def FD_Kitti_get_train_dataset(inp_dir):
  return Kitti_FDTrainLoadDataset(inp_dir)

def FD_Kitti_get_val_test_dataset(inp_dir):
  return Kitti_FDValTestLoadDataset(inp_dir)

def SEG_get_train_dataset(inp_dir, tar_dir, patch_size):
  return SegTrainLoadDataset(inp_dir, tar_dir, patch_size)

def SEG_get_val_test_dataset(inp_dir, tar_dir):
  return SegValTestLoadDataset(inp_dir, tar_dir)

def SEG_Kitti_get_train_dataset(inp_dir, tar_dir, patch_size):
  return SegKittiTrainLoadDataset(inp_dir, tar_dir, patch_size)

def SEG_Kitti_get_val_test_dataset(inp_dir, tar_dir):
  return SegKittiValTestLoadDataset(inp_dir, tar_dir)