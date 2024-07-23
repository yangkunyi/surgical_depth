import torch
import torch.nn as nn
from torchvision.models import resnet18  # 或者其他 ResNet 变体

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 提取 ResNet 中的层
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 调整通道数以匹配您的输出要求

        self.conv_out1 = nn.Conv3d(64, 96, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.conv_out2 = nn.Conv3d(128, 192, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.conv_out3 = nn.Conv3d(256, 384, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.conv_out4 = nn.Conv3d(512, 768, kernel_size=(1, 1, 1), stride=1, padding=0)

    def forward(self, x):
        # 在 depth 维度上迭代，对每一帧应用 ResNet
        outs = []
        for i in range(x.shape[2]): 
            frame = x[:, :, i, :, :]

            frame = self.conv1(frame)
            frame = self.bn1(frame)
            frame = self.relu(frame)
            # frame = self.maxpool(frame)

            out1 = self.layer1(frame)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)

            outs.append((out1, out2, out3, out4))

        # 将每一帧的输出在 depth 维度上堆叠
        out1 = torch.stack([o[0] for o in outs], dim=2)
        out2 = torch.stack([o[1] for o in outs], dim=2)
        out3 = torch.stack([o[2] for o in outs], dim=2)
        out4 = torch.stack([o[3] for o in outs], dim=2)

        # 使用 3D 卷积调整通道数
        out1 = self.conv_out1(out1) 
        out2 = self.conv_out2(out2) 
        out3 = self.conv_out3(out3) 
        out4 = self.conv_out4(out4) 

        return out1, out2, out3, out4