import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
from customdataset.get_load_dataset import *
import time
from tqdm import tqdm
from losses.losses import *
from models.flare_detect_cam import *
import pandas as pd

class Trainier(nn.Module):
    def __init__(self, save_model_weights_path, device, lr, num_epochs):
        super(Trainier, self).__init__()
        
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        model = Net()

        self.model = model.to(device)
        
        save_root = '/workspace'
        model_weights_save = os.path.join(save_root, save_model_weights_path)
        os.makedirs(model_weights_save, exist_ok=True)
        
        self.model_weights_save = model_weights_save

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        self.MLS_loss = nn.MultiLabelSoftMarginLoss().to(self.device)

        self.Training_history = {'loss': []}
        self.Validation_history = {'loss': []}

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}/{} Start......".format(epoch, self.num_epochs))
            epoch_loss = self._train_epoch(epoch, train_loader)
            epoch_val_loss = self._valid_epoch(epoch, val_loader)

        self.Training_history = pd.DataFrame.from_dict(self.Training_history, orient='index')
        self.Training_history.to_csv(os.path.join(self.model_weights_save, 'train_history.csv'))
        self.Validation_history = pd.DataFrame.from_dict(self.Validation_history, orient='index')
        self.Validation_history.to_csv(os.path.join(self.model_weights_save, 'valid_history.csv'))

        print('Best loss index:', self.Validation_history.loc['loss'].idxmin() + 1, 'Best loss: ', self.Validation_history.loc['loss'].min())

    def _train_epoch(self, epoch, train_loader):
        epoch_start_time = time.time()
        
        self.model.train()
        
        epoch_loss = 0
        print("======> Train Start")
        for i, data in enumerate(tqdm(train_loader), 0):
            input = data[0].to(self.device)
            target = data[1].to(self.device)

            x = self.model(input)
            loss = self.MLS_loss(x, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        
        print('Epoch: {}\tTime: {:.4f}\tLoss: {:.8f}\tLR: {:.8f}'.format(
            epoch, time.time() - epoch_start_time, epoch_loss / len(train_loader), self.lr))
        
        self.Training_history['loss'].append(epoch_loss / len(train_loader))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'gen_optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))

        return epoch_loss

    def _valid_epoch(self, epoch, val_loader):
        self.model.eval()
        with torch.no_grad():
            epoch_start_time = time.time()
            epoch_val_loss = 0
            print('======> Validation Start')
            for i, data in enumerate(tqdm(val_loader)):
                val_input = data[0].to(self.device)
                val_target = data[1].to(self.device)

                val_x = self.model(val_input)
                val_loss = self.MLS_loss(val_x, val_target)

                epoch_val_loss += val_loss.item()

            print('Epoch: {}\tTime: {:.4f}\tLoss: {:.8f}'.format(epoch, time.time() - epoch_start_time, epoch_val_loss / len(val_loader)))

            self.Validation_history['loss'].append(epoch_val_loss / len(val_loader))

        return epoch_val_loss

