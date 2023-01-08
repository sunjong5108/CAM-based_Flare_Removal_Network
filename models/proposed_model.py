'''
CAM-FRN: Class-attention map-based Flare Removal Network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

'''
1. Module 1. Atrous convolution Dual Channel Attention Residual Block (ADCARB)
We modified CycleGAN Residual Block inspired by AOT-GAN's AOT-block
'''
class CALayer(nn.Module):
    def __init__(self, channel):
        '''
        We replaced ReLU to GELU
        '''
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class ADCARB(nn.Module):
    def __init__(self, dim):
        super(ADCARB, self).__init__()

        self.conv_dilated_1 = nn.Conv2d(dim, dim//16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='reflect')
        self.conv_dilated_4 = nn.Conv2d(dim, dim//16, kernel_size=3, stride=1, padding=4, dilation=4, bias=True, padding_mode='reflect')
        self.conv_dilated_16 = nn.Conv2d(dim, dim//16, kernel_size=3, stride=1, padding=16, dilation=16, bias=True, padding_mode='reflect')
        self.conv3x3= nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
        self.conv_fuse= nn.Conv2d(3*(dim//16), dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
        self.conv_dest = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
        self.activation = nn.GELU()
        self.calayer = CALayer(dim)
        self.InstanceNorm = nn.InstanceNorm2d(dim, affine=False)

        self._init_weight()

    def forward(self, x):
        x_di1 = self.activation(self.conv_dilated_1(x))
        x_di4 = self.activation(self.conv_dilated_4(x))
        x_di16 = self.activation(self.conv_dilated_16(x))

        mask = self.calayer(self.conv_dest(x))
        mask = torch.sigmoid(mask)

        x_c = self.conv_fuse(torch.cat((x_di1, x_di4, x_di16), dim=1))

        x_c = self.activation(self.InstanceNorm(x_c))
        x_c = self.InstanceNorm(self.conv3x3(x_c))

        x_c = x * (1 - mask) + x_c * mask
        
        x_c = self.calayer(x_c)
        x_c += x
        
        return x_c

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.InstanceNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
'''
2. Using self-attention proposed by "Stand-Alone Self-Attention in Vision Models" 
'''
class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

'''
3. CAM-FRN Encoder
'''
class ResidualModule(nn.Module):
    def __init__(self, dim, kernel_size, bias): 
        super(ResidualModule, self).__init__()

        self.conv1 = nn.Conv2d(dim, 3, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=bias, padding_mode='reflect')
      
        self._init_weight()

    def forward(self, dec, x_img):
        # _, _, h, w = x_img.size()

        # _, _, oh, ow = dec.size()

        # if h != oh or w != ow or (h != oh and w != ow): 
        #     dec = F.interpolate(dec, size=(h, w), mode='bilinear')
        # else:
        #     dec = dec

        out = self.conv1(dec)

        img = out + x_img

        return img
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, ch, block, blocks, dim, z_dim):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(ch, dim, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='reflect')
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='reflect')
        modules = [block(dim) for _ in range(blocks)]
        self.ResBlock = nn.Sequential(*modules)

        self.conv_mu = nn.Conv2d(dim, z_dim, kernel_size=3, stride=1, padding=1)
        self.conv_log_var = nn.Conv2d(dim, z_dim, kernel_size=3, stride=1, padding=1)

        self._init_weight()

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ResBlock(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        z = self.sample(mu, log_var)

        return z, x, mu, log_var
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

'''
4. CAM-FRN Decoder
'''
class Decoder(nn.Module):
    def __init__(self, block, blocks, dim, z_dim):
        super(Decoder, self).__init__()

        modules = [block(z_dim) for _ in range(blocks)]
        self.ResBlock = nn.Sequential(*modules)
        self.deconv3 = nn.ConvTranspose2d(z_dim, dim, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv4 = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=True)
       
        self._init_weight()

    def forward(self, z):
        z = self.ResBlock(z)
        z = self.deconv3(z)
        z_deconv4 = self.deconv4(z)

        return z_deconv4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

'''
5. CAM-FRN
'''
class CAM_FRN(nn.Module):
    def __init__(self, ch, blocks, dim, z_dim):
        super(CAM_FRN, self).__init__()

        self.encoder = Encoder(ch+3+3+1, ADCARB, blocks, dim, z_dim)
        self.attention_conv = AttentionStem(z_dim, z_dim, 3, stride=1, padding=1, groups=8, m=4, bias=True)
        self.ff_conv = nn.Conv2d(2*z_dim, z_dim, kernel_size=3, padding=1, stride=1, padding_mode='reflect')
        self.decoder = Decoder(ADCARB, blocks, dim, z_dim)
        self.rm = ResidualModule(dim, 3, True)

    def forward(self, x, cam, masked_input, mask): #x, cam, degradation, mask
        x_clone = x.clone()
        x_clone = x_clone * cam # class attention map multiply
        gen_input = torch.cat((x, x_clone, masked_input, mask), dim=1)

        encoding, f_x, mu, log_var = self.encoder(gen_input)
        encoding_att = self.attention_conv(encoding)
        encoding_ = encoding
        fuse_feature = self.ff_conv(torch.cat((encoding_, encoding_att), dim=1))

        dec = self.decoder(fuse_feature)
        final_recon = self.rm(dec, x)
        
        final_recon = torch.sigmoid(final_recon)

        return final_recon, encoding, mu, log_var

    def get_inference_lr_params(self):
        modules = [self.encoder, self.attention_conv, self.ff_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_generator_lr_params(self):
        modules = [self.decoder, self.rm]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.ConvTranspose2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

'''
6. Discriminator in PatchGAN
'''

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        self.use_sigmoid = use_sigmoid

        self.conv1_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._init_weight()

    def forward(self, x):
        conv1_out = self.conv1_block(x)
        conv2_out = self.conv2_block(conv1_out)
        conv3_out = self.conv3_block(conv2_out)
        conv4_out = self.conv4_block(conv3_out)
        conv5_out = self.conv5_block(conv4_out)
        
        output = conv5_out
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == '__main__':
    # from thop import profile
    # model = CAM_FRN(3, 5, 128, 512)
    
    # #dis = Discriminator(3)
    # model.eval()

    # input = torch.rand(1, 3, 512, 512)
    # highres_cam = torch.rand(1, 1, 512, 512)
    # fr_masked_image = torch.rand(1, 3, 512, 512)
    # m = torch.rand(1, 1, 512, 512)

    # final_out, encoding, mu, logvar = model(input, highres_cam, fr_masked_image, m)
    # #dis_out = dis(final_out)
    # macs, params = profile(model, inputs=(input, highres_cam, fr_masked_image, m))
    # #dis_macs, dis_params = profile(dis, inputs=(final_out))


    # print('macs', macs)
    # print('flops', (macs)/2)
    # print('params', params)
    # print('Output size', final_out.size())

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    # timings = np.zeros((repetitions, 1))

    # # for _ in range(10):
    # #     _ = model(dummy_input)

    # model_ = CAM_FRN(3, 5, 128, 512).cuda()
    # #dummy_input = input
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model_(input.cuda(), highres_cam.cuda(), fr_masked_image.cuda(), m.cuda())
    #         ender.record()

    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    
    # print(timings)
    # print(np.mean(timings))

    from torchinfo import summary

    input = torch.rand(1, 3, 300, 300)

    dis_model = Discriminator(3)
    
    #dis = Discriminator(3)
    dis_model.eval()

    summary(Discriminator(3), input_size=(1, 3, 300, 300))