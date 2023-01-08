'''ResNet-50 for flare or clean classification https://github.com/jiwoon-ahn/irn/blob/master/net/resnet50.py '''
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

model_weights_url = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, bottleneck, layers, strides=[2, 2, 2, 2], dilations=[1, 1, 1, 1]):
        super(ResNet50, self).__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(bottleneck, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(bottleneck, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(bottleneck, 256, layers[2], stride=strides[3], dilation=dilations[2])
        self.layer4 = self._make_layer(bottleneck, 512, layers[3], stride=strides[2], dilation=dilations[3])
        self.in_ch = 1024

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * 4, 1000)

    def _make_layer(self, bottleneck, ch, block_num, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_ch != ch * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, ch * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch * 4)
            )
        
        layers = [bottleneck(self.in_ch, ch, stride, downsample, dilation=1)]
        self.in_ch = ch * 4
        for _ in range(1, block_num):
            layers.append(bottleneck(self.in_ch, ch, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=True, strides=[2, 2, 2, 2], dilations=[1, 1, 1, 1]):
    model = ResNet50(BottleNeck, layers=[3, 4, 6, 3], strides=strides, dilations=dilations)
    if pretrained:
        state_dict = model_zoo.load_url(model_weights_url['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)

    return model

'''https://github.com/jiwoon-ahn/irn/blob/master/net/resnet50_cam.py'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 2, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 2)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        return x