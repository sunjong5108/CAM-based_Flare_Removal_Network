import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import numpy as np
from customdataset.get_load_dataset import *
import time
from tqdm import tqdm
from losses.losses import *
from models.proposed_model import *
from models.flare_detect_cam import *
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from utils.inception import *
from utils.metrics import *
from utils.LearningScheduler import *
import pandas as pd

class Trainier(nn.Module):
    def __init__(self, save_root, save_model_weights_path, save_val_results_path, FD_model_weights_path,  device, patch_size, lr, num_epochs): 
        super(Trainier, self).__init__()

        self.ps = patch_size
        self.beta = 1

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        incepv3 = InceptionV3([block_idx])
        self.incepv3 = incepv3.to(device)

        vae = CAM_FRN(ch=3, blocks=5, dim=128, z_dim=512)
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)

        LensFlareMask_extractor = CAM()
        checkpoint = torch.load(FD_model_weights_path)
        LensFlareMask_extractor.load_state_dict(checkpoint['model_state_dict'])
        print('flare CAM checkpoint loading complete...')

        for target_param_cam in LensFlareMask_extractor.parameters():
            target_param_cam.requires_grad = False

        self.lensflaremask_est = LensFlareMask_extractor.to(device)
        self.gen_model = vae.to(device)
        self.dis_model = discriminator.to(device)

        model_weights_save = os.path.join(save_root, save_model_weights_path)
        os.makedirs(model_weights_save, exist_ok=True)

        result_save = os.path.join(save_root, save_val_results_path)
        os.makedirs(result_save, exist_ok=True)
        
        self.model_weights_save = model_weights_save
        self.save_val_results_path = result_save

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs

        self.scale = [1.0, 0.5, 1.5, 2.0]
        
        train_params = [{'params': self.gen_model.get_inference_lr_params(), 'lr': self.lr * 1},
                        {'params': self.gen_model.get_generator_lr_params(), 'lr': self.lr * 1}]

        self.gen_optimizer = optim.Adam(train_params, lr=self.lr)
        self.dis_optimizer = optim.Adam(self.dis_model.parameters(), lr=self.lr*0.1)

        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.style_loss = StyleLoss().to(self.device)
        self.l2_loss = nn.MSELoss().to(self.device)
        self.edge_loss = EdgeLoss().to(self.device)

        self.Training_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': []}
        self.Validation_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': [], 'fid': []}

    def eval_step(self, engine, batch):
        return batch

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}/{} Start......".format(epoch, self.num_epochs))
            epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim = self._train_epoch(epoch, train_loader)
            epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim = self._valid_epoch(epoch, val_loader)

        self.Training_history = pd.DataFrame.from_dict(self.Training_history, orient='index')
        self.Training_history.to_csv(os.path.join(self.model_weights_save, 'train_history.csv'))
        self.Validation_history = pd.DataFrame.from_dict(self.Validation_history, orient='index')
        self.Validation_history.to_csv(os.path.join(self.model_weights_save, 'valid_history.csv'))

        print('Best PSNR score index:', self.Validation_history.loc['psnr'].idxmax() + 1, 'Best PSNR score:', self.Validation_history.loc['psnr'].max())
        print('Best SSIM score index:', self.Validation_history.loc['ssim'].idxmax() + 1, 'Best SSIM score:', self.Validation_history.loc['ssim'].max())
        print('Best FID score index:', self.Validation_history.loc['fid'].idxmin() + 1, 'Best FID score:', self.Validation_history.loc['fid'].min())

    def _train_epoch(self, epoch, train_loader):
        ps = self.ps
        epoch_start_time = time.time()
        default_evaluator = Engine(self.eval_step)

        metric_ssim = SSIM(1.0)
        metric_ssim.attach(default_evaluator, 'ssim')

        metric_psnr = PSNR(1.0)
        metric_psnr.attach(default_evaluator, 'psnr')
        
        self.gen_model.train()
        self.dis_model.train()
        
        epoch_dis_loss = 0
        epoch_dis_real_loss = 0
        epoch_dis_fake_loss = 0
        epoch_dis_masked_real_loss = 0
        epoch_dis_masked_fake_loss = 0
        epoch_gen_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_tv_loss = 0
        
        print("======> Train Start")
        for i, data in enumerate(tqdm(train_loader), 0):
            input = data[0].to(self.device)
            clean = data[1].to(self.device)

            fmd_inputs_list = []
            size = (input.shape[2], input.shape[3]) # w, h
            for s in self.scale:
                target_size = (int(np.round(size[0]*s)), int(np.round(size[1]*s)))
                fmd_inputs_list.append(F.interpolate(input, size=target_size, mode='bicubic'))
            
            fmd_outputs = [self.lensflaremask_est(img) for img in fmd_inputs_list]

            highres_cam = torch.sum(torch.stack([F.interpolate(o, size, mode='bilinear', align_corners=False) for o in fmd_outputs]), 0)
            
            highres_cam = highres_cam[:, 0, :, :].unsqueeze(1)
            highres_cam /= F.adaptive_avg_pool2d(highres_cam, (1, 1)) + 1e-5
            
            m = highres_cam.clone()
            m[m >= 0.2] = 1
            m[m < 0.2] = 0

            fr_masked_image = input * (1 - m) + m

            w = input.shape[2] 
            h = input.shape[3]

            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
            
            if padw!=0 or padh!=0:
                input = TF.pad(input, (0,0,padw,padh), padding_mode='reflect')
                clean = TF.pad(clean, (0,0,padw,padh), padding_mode='reflect')
                m = TF.pad(m, (0,0,padw,padh), padding_mode='reflect')
                fr_masked_image = TF.pad(fr_masked_image, (0,0,padw,padh), padding_mode='reflect')
                highres_cam  = TF.pad(highres_cam, (0,0,padw,padh), padding_mode='reflect')
            
            hh, ww = clean.shape[2], clean.shape[3]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)

            input = input[:, :, rr:rr+ps, cc:cc+ps]
            highres_cam =highres_cam[:, :, rr:rr+ps, cc:cc+ps]
            clean = clean[:, :, rr:rr+ps, cc:cc+ps]
            m = m[:, :, rr:rr+ps, cc:cc+ps]
            fr_masked_image = fr_masked_image[:, :, rr:rr+ps, cc:cc+ps]
        
            final_out, encoding, mu, logvar = self.gen_model(input, highres_cam, fr_masked_image, m)

            # Discriminator
            self.dis_optimizer.zero_grad()

            dis_real, _ = self.dis_model(clean)
            dis_fake, _ = self.dis_model(final_out.detach())

            masked_dis_real, _ = self.dis_model(clean * m)
            masked_dis_fake, _ = self.dis_model(final_out.detach() * m)


            dis_real_loss = self.l2_loss(dis_real, torch.ones_like(dis_real))
            dis_fake_loss = self.l2_loss(dis_fake, torch.zeros_like(dis_fake))

            masked_dis_real_loss = self.l2_loss(masked_dis_real, torch.ones_like(masked_dis_real))
            masked_dis_fake_loss = self.l2_loss(masked_dis_fake, torch.zeros_like(masked_dis_fake))

            dis_loss = (dis_real_loss + dis_fake_loss) / 2 + (masked_dis_real_loss + masked_dis_fake_loss) / 2

            dis_loss.backward()
            self.dis_optimizer.step()

            epoch_dis_loss += dis_loss.item()
            epoch_dis_real_loss += dis_real_loss.item()
            epoch_dis_fake_loss += dis_fake_loss.item()
            epoch_dis_masked_real_loss += masked_dis_real_loss.item()
            epoch_dis_masked_fake_loss += masked_dis_fake_loss.item()

            # Generator
            self.gen_optimizer.zero_grad()

            vae_loss_ = vae_loss(final_out, clean, mu, logvar, beta=self.beta)
            vae_loss_ += self.perceptual_loss(final_out, clean)

            gen_fake, _ = self.dis_model(final_out)
            gen_gan_loss = self.l2_loss(gen_fake, torch.ones_like(gen_fake)) * 0.01

            masked_gen_fake, _ = self.dis_model(final_out * m)
            masked_gen_gan_loss = self.l2_loss(masked_gen_fake, torch.ones_like(masked_gen_fake)) * 0.01
            
            gen_style_loss = self.style_loss(final_out, clean)

            masked_gen_style_loss = self.style_loss(final_out*m, clean*m)
            masked_gen_per_loss = self.perceptual_loss(final_out*m, clean*m)

            gen_edge_loss = self.edge_loss(final_out, clean)

            gen_loss = vae_loss_ + (gen_gan_loss + masked_gen_gan_loss) + 30*(gen_style_loss + masked_gen_style_loss) + masked_gen_per_loss + gen_edge_loss + tv_loss(final_out, 2e-6)

            gen_loss.backward()
            self.gen_optimizer.step()

            epoch_gen_loss += gen_loss.item()

            metric_state = default_evaluator.run([[final_out, clean]])

            epoch_psnr += metric_state.metrics['psnr']
            epoch_ssim += metric_state.metrics['ssim']
            epoch_tv_loss += tv_loss(final_out, 2e-6)
        
        print('Epoch: {}\tTime: {:.4f}\tBeta: {}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tGen LR: {:.8f}\tDis LR: {:.8f}'.format(
                epoch, time.time() - epoch_start_time, self.beta, epoch_gen_loss / len(train_loader), epoch_dis_loss / len(train_loader), 
                epoch_psnr / len(train_loader), epoch_ssim / len(train_loader),
                self.lr, self.lr*0.1))
        print('Dis real Loss: {:.4f}\tDis fake Loss: {:.4f}\tDis masked real Loss: {:.4f}\tDis masked fake Loss: {:.4f}'.format(
            epoch_dis_real_loss / len(train_loader), epoch_dis_fake_loss / len(train_loader), epoch_dis_masked_real_loss / len(train_loader), epoch_dis_masked_fake_loss / len(train_loader)
        ))
        print('TV_loss: {}'.format(epoch_tv_loss / len(train_loader)))
        self.Training_history['gen_loss'].append(epoch_gen_loss / len(train_loader))
        self.Training_history['dis_loss'].append(epoch_dis_loss / len(train_loader))
        self.Training_history['psnr'].append(epoch_psnr / len(train_loader))
        self.Training_history['ssim'].append(epoch_ssim / len(train_loader))

        torch.save({
            'epoch': epoch,
            'vae_state_dict': self.gen_model.state_dict(),
            'discriminator_state_dict': self.dis_model.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict()
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))

        return epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim

    def _valid_epoch(self, epoch, val_loader):
        self.gen_model.eval()
        self.dis_model.eval()
        with torch.no_grad():
            val_epoch_start_time = time.time()
            val_default_evaluator = Engine(self.eval_step)

            val_metric_ssim = SSIM(1.0)
            val_metric_ssim.attach(val_default_evaluator, 'ssim')

            val_metric_psnr = PSNR(1.0) 
            val_metric_psnr.attach(val_default_evaluator, 'psnr')

            epoch_val_dis_loss = 0
            epoch_val_gen_loss = 0
            epoch_val_psnr = 0
            epoch_val_ssim = 0
            epoch_val_fid = 0
            epoch_tv_loss = 0

            print('======> Validation Start')
            for i, data in enumerate(tqdm(val_loader)):
                val_input = data[0].to(self.device)
                val_clean = data[1].to(self.device)

                val_fmd_inputs_list = []
                size = (val_input.shape[2], val_clean.shape[3])
                for s in self.scale:
                    target_size = (int(np.round(size[0]*s)), int(np.round(size[1]*s)))
                    val_fmd_inputs_list.append(F.interpolate(val_input, size=target_size, mode='bicubic'))
                
                val_fmd_outputs = [self.lensflaremask_est(img) for img in val_fmd_inputs_list]

                val_highres_cam = torch.sum(torch.stack([F.interpolate(o, self.strided_up_size, mode='bilinear', align_corners=False) for o in val_fmd_outputs]), 0)
                
                val_highres_cam = val_highres_cam[:, 0, :, :].unsqueeze(1)
                val_highres_cam /= F.adaptive_avg_pool2d(val_highres_cam, (1, 1)) + 1e-5
                
                val_m = val_highres_cam.clone()
                val_m[val_m >= 0.2] = 1
                val_m[val_m < 0.2] = 0

                val_fr_masked_image = val_input * (1 - val_m) + val_m
                
                val_final_out, val_encoding, val_mu, val_logvar = self.gen_model(val_input, val_highres_cam, val_fr_masked_image, val_m)

                val_dis_real, _ = self.dis_model(val_clean)
                val_dis_fake, _ = self.dis_model(val_final_out.detach())

                val_masked_dis_real, _ = self.dis_model(val_clean * val_m)
                val_masked_dis_fake, _ = self.dis_model(val_final_out.detach() * val_m)

                val_dis_real_loss = self.l2_loss(val_dis_real, torch.ones_like(val_dis_real))
                val_dis_fake_loss = self.l2_loss(val_dis_fake, torch.zeros_like(val_dis_fake))

                val_masked_dis_real_loss = self.l2_loss(val_masked_dis_real, torch.ones_like(val_masked_dis_real))
                val_masked_dis_fake_loss = self.l2_loss(val_masked_dis_fake, torch.zeros_like(val_masked_dis_fake))

                val_dis_loss = (val_dis_real_loss + val_dis_fake_loss) / 2 + (val_masked_dis_real_loss + val_masked_dis_fake_loss) / 2
                
                epoch_val_dis_loss += val_dis_loss.item()

                # Generator

                val_vae_loss_ = vae_loss(val_final_out, val_clean, val_mu, val_logvar, beta=self.beta)
                val_vae_loss_ += self.perceptual_loss(val_final_out, val_clean)

                val_gen_fake, _ = self.dis_model(val_final_out)
                val_gen_gan_loss = self.l2_loss(val_gen_fake, torch.ones_like(val_gen_fake)) * 0.01

                val_masked_gen_fake, _ = self.dis_model(val_final_out * val_m)
                val_masked_gen_gan_loss = self.l2_loss(val_masked_gen_fake, torch.ones_like(val_masked_gen_fake)) * 0.01

                val_gen_style_loss = self.style_loss(val_final_out, val_clean)
                val_gen_edge_loss = self.edge_loss(val_final_out, val_clean)

                val_masked_gen_style_loss = self.style_loss(val_final_out*val_m, val_clean*val_m)
                val_masked_gen_per_loss = self.perceptual_loss(val_final_out*val_m, val_clean*val_m)

                val_gen_loss = val_vae_loss_ + (val_gen_gan_loss + val_masked_gen_gan_loss) + 30*(val_gen_style_loss + val_masked_gen_style_loss) + val_masked_gen_per_loss + val_gen_edge_loss + tv_loss(val_final_out, 2e-6)

                epoch_val_gen_loss += val_gen_loss.item()

                val_metric_state = val_default_evaluator.run([[val_final_out, val_clean]])
                val_fid = calculate_fretchet(val_final_out, val_clean, self.incepv3)
    
                epoch_val_psnr += val_metric_state.metrics['psnr']
                epoch_val_ssim += val_metric_state.metrics['ssim']
                epoch_val_fid += val_fid
                epoch_tv_loss += tv_loss(val_final_out, 2e-6)

                torchvision.utils.save_image(torch.cat([val_input, val_final_out, val_clean]), os.path.join(self.save_val_results_path, 'results_' + str(epoch) + '.png'))
            print('Epoch: {}\tTime: {:.4f}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tFID: {:.4f}'.format(
                epoch, time.time() - val_epoch_start_time, epoch_val_gen_loss / len(val_loader), epoch_val_dis_loss / len(val_loader), 
                epoch_val_psnr / len(val_loader), epoch_val_ssim / len(val_loader), epoch_val_fid / len(val_loader)))
            print('TV_loss: {}'.format(epoch_tv_loss / len(val_loader)))
            self.Validation_history['gen_loss'].append(epoch_val_gen_loss / len(val_loader))
            self.Validation_history['dis_loss'].append(epoch_val_dis_loss / len(val_loader))
            self.Validation_history['psnr'].append(epoch_val_psnr / len(val_loader))
            self.Validation_history['ssim'].append(epoch_val_ssim / len(val_loader))
            self.Validation_history['fid'].append(epoch_val_fid / len(val_loader))

        return epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim

