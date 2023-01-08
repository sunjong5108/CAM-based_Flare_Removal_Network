import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from tqdm import tqdm
from segmodel.DeepLabV3_plus import *
from utils.LearningScheduler import *
from utils.metrics import *
from utils.utils_func import *
import pandas as pd

class Trainier(nn.Module):
    def __init__(self, train_loader, label_color_map, save_model_weights_path, device, lr, num_epochs): 
        super(Trainier, self).__init__()

        self.seg_model = DeepLab(output_stride=8, num_classes=12).to(device)
        
        save_root = '/workspace'
        model_weights_save = os.path.join(save_root, save_model_weights_path)
        os.makedirs(model_weights_save, exist_ok=True)
        
        self.model_weights_save = model_weights_save

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.class_nums = len(label_color_map)
        self.class_weights = calc_class_weights(train_loader, self.class_nums)

        train_params = [{'params': self.seg_model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.seg_model.get_10x_lr_params(), 'lr': self.lr * 10}]

        self.optimizer = optim.Adam(train_params, lr=self.lr)

        self.lr_scheduler = Poly_Warm_Cos_LR(self.optimizer, 
                                            warm_up_steps=250,
                                            T_max=50, eta_min=1e-6, style='floor_4', power=3.0, max_epoch=self.num_epochs)

        self.WCE = nn.CrossEntropyLoss(weight=torch.from_numpy(self.class_weights.astype(np.float32)), ignore_index=-1)
        
        self.Training_history = {'seg_loss': [], 'miou': []}
        self.Validation_history = {'seg_loss': [], 'miou': []}
        self.Test_history = {'seg_loss': [], 'miou': []}

    def train(self, train_loader, val_loader, test_loader, train_batch, eval_batch):
        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}/{} Start......".format(epoch, self.num_epochs))
            self._train_epoch(epoch, train_loader, train_batch)
            self._valid_epoch(epoch, val_loader, eval_batch)
            self._test_epoch(epoch, test_loader, eval_batch)

        self.Training_history = pd.DataFrame.from_dict(self.Training_history, orient='index')
        self.Training_history.to_csv(os.path.join(self.model_weights_save, 'train_history.csv'))
        self.Validation_history = pd.DataFrame.from_dict(self.Validation_history, orient='index')
        self.Validation_history.to_csv(os.path.join(self.model_weights_save, 'valid_history.csv'))
        self.Test_history = pd.DataFrame.from_dict(self.Test_history, orient='index')
        self.Test_history.to_csv(os.path.join(self.model_weights_save, 'test_history.csv'))

        print('Validation')
        print('Best MIOU score index:', self.Validation_history.loc['miou'].idxmax() + 1, 'Best PSNR score:', self.Validation_history.loc['miou'].max())
        print('Test')
        print('Best MIOU score index:', self.Test_history.loc['miou'].idxmax() + 1, 'Best PSNR score:', self.Test_history.loc['miou'].max())

    def _train_epoch(self, epoch, train_loader, batch_size):
        epoch_start_time = time.time()
        self.seg_model.train()

        epoch_seg_loss = 0

        total_batch = len(train_loader)
        total_class_ious = np.zeros((total_batch * batch_size, self.class_nums))

        print("======> Train Start")
        for i, data in enumerate(tqdm(train_loader), 0):
            batch_ious_class = np.zeros((batch_size, self.class_nums))
            input = data[0].to(self.device)
            label = data[1].to(self.device)
     
            target = label.long()
            target[target == 11] = -1
            final_out = self.seg_model(input)

            seg_loss = self.WCE(final_out, target)

            self.optimizer.zero_grad()
            seg_loss.backward()
            self.optimizer.step()

            epoch_seg_loss += seg_loss.item()

            for j in range(batch_size):
                batch_ious_class[j] = iou_calc(torch.argmax(final_out.to('cpu'), dim=1)[j], label[j].to('cpu'), void=True, class_num=self.class_nums)

            total_class_ious[i*batch_size:(i+1)*batch_size, :] = batch_ious_class[:, :]
        
        self.lr_scheduler.step()

        class_ious_per_epoch = np.nanmean(total_class_ious, axis=0)
        epoch_miou = np.nanmean(class_ious_per_epoch, axis=0)

        print('Epoch: {}\tTime: {:.4f}\tSeg Loss: {:.4f}\tMIoU: {:.4f}\tLR: {:.8f}'.format(
                epoch, time.time() - epoch_start_time, epoch_seg_loss / len(train_loader), epoch_miou,
                self.lr_scheduler.get_last_lr()[0]))

        self.Training_history['seg_loss'].append(epoch_seg_loss / len(train_loader))
        self.Training_history['miou'].append(epoch_miou)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.seg_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))

    def _valid_epoch(self, epoch, val_loader, batch_size):
        epoch_start_time = time.time()
        self.seg_model.eval()
        with torch.no_grad():
            val_epoch_seg_loss = 0

            val_total_batch = len(val_loader)
            val_total_class_ious = np.zeros((val_total_batch * batch_size, self.class_nums))

            print("======> Validation Start")
            for i, data in enumerate(tqdm(val_loader), 0):
                val_batch_ious_class = np.zeros((batch_size, self.class_nums))
                val_input = data[0].to(self.device)
                val_label = data[1].to(self.device)

                val_target = val_label.long()
                val_target[val_target == 11] = -1

                val_final_out = self.seg_model(val_input)

                val_seg_loss = self.WCE(val_final_out, val_target)

                val_epoch_seg_loss += val_seg_loss.item()

                for j in range(batch_size):
                    val_batch_ious_class[j] = iou_calc(torch.argmax(val_final_out.to('cpu'), dim=1)[j], val_label[j].to('cpu'), void=True, class_num=self.class_nums)

                val_total_class_ious[i*batch_size:(i+1)*batch_size, :] = val_batch_ious_class[:, :]

            val_class_ious_per_epoch = np.nanmean(val_total_class_ious, axis=0)
            val_epoch_miou = np.nanmean(val_class_ious_per_epoch, axis=0)

            print('Epoch: {}\tTime: {:.4f}\tSeg Loss: {:.4f}\tMIoU: {:.4f}\tLR: {:.8f}'.format(
                    epoch, time.time() - epoch_start_time, val_epoch_seg_loss / len(val_loader), val_epoch_miou,
                    self.lr_scheduler.get_last_lr()[0]))

            self.Validation_history['seg_loss'].append(val_epoch_seg_loss / len(val_loader))
            self.Validation_history['miou'].append(val_epoch_miou)
    
    def _test_epoch(self, epoch, test_loader, batch_size):
        epoch_start_time = time.time()
        self.seg_model.eval()
        with torch.no_grad():
            test_epoch_seg_loss = 0

            test_total_batch = len(test_loader)
            test_total_class_ious = np.zeros((test_total_batch * batch_size, self.class_nums))

            print("======> Test Start")
            for i, data in enumerate(tqdm(test_loader), 0):
                test_batch_ious_class = np.zeros((batch_size, self.class_nums))
                test_input = data[0].to(self.device)
                test_label = data[1].to(self.device)

                test_target = test_label.long()
                test_target[test_target == 11] = -1
                
                test_final_out = self.seg_model(test_input)

                test_seg_loss = self.WCE(test_final_out, test_target)

                test_epoch_seg_loss += test_seg_loss.item()

                for j in range(batch_size):
                    test_batch_ious_class[j] = iou_calc(torch.argmax(test_final_out.to('cpu'), dim=1)[j], test_label[j].to('cpu'), void=True, class_num=self.class_nums)

                test_total_class_ious[i*batch_size:(i+1)*batch_size, :] = test_batch_ious_class[:, :]

            test_class_ious_per_epoch = np.nanmean(test_total_class_ious, axis=0)
            test_epoch_miou = np.nanmean(test_class_ious_per_epoch, axis=0)

            print('Epoch: {}\tTime: {:.4f}\tSeg Loss: {:.4f}\tMIoU: {:.4f}\tLR: {:.8f}'.format(
                    epoch, time.time() - epoch_start_time, test_epoch_seg_loss / len(test_loader), test_epoch_miou,
                    self.lr_scheduler.get_last_lr()[0]))

            self.Test_history['seg_loss'].append(test_epoch_seg_loss / len(test_loader))
            self.Test_history['miou'].append(test_epoch_miou)
        

