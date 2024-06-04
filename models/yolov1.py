#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

from torchvision.models.resnet import Bottleneck
import math
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models.resnet
import torch


class YOLOv1(torchvision.models.resnet.ResNet):

    def __init__(self, block, layers, num_classes=20, num_bboxes=2):
        super(YOLOv1, self).__init__(block, layers)

        self.B = num_bboxes
        self.C = num_classes

        self.layer5 = Bottleneck(inplanes=2048,planes=512,stride=1)

        self.end = nn.Sequential(
            nn.Conv2d(2048, self.C + self.B * 5, 3,stride=2, padding=1),
            nn.BatchNorm2d(self.C + self.B * 5),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.end(x)

        x = x.permute(0, 2, 3, 1)

        return x


def YOLO_v1(num_classes=20, num_bboxes=2):
    model = YOLOv1(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, num_bboxes = num_bboxes)

    return model