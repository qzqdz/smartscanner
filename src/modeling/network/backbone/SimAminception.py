from torch import nn
from torchvision import models

from src.utils.registry import REGISTRY


# Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
import torch
import torch.nn as nn
from thop import profile



class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)




@REGISTRY.register('simaninception')
class SimAmInceptionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SimAmInceptionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.AuxLogits.fc = nn.Linear(768, num_classes)
        self.inception.fc = nn.Linear(2048, num_classes)
        self.simam = Simam_module()

    def forward(self, inputs):
        # Inception模型的辅助输出仅在训练时使用
        if self.training:
            outputs, aux_outputs = self.inception(inputs)
            return self.simam(outputs), self.simam(aux_outputs)
        else:
            outputs = self.inception(inputs)
            return self.simam(outputs)
    
    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' in param_tuple[0], self.inception.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' not in param_tuple[0], self.inception.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups


@REGISTRY.register('simam_resnet')
class ResNetModel(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # 创建 Simam_module 实例
        self.simam = Simam_module()

        # 替换全连接层以适应分类任务
        if classify:
            self.resnet.fc = nn.Linear(512, num_classes)
        else:
            features = nn.ModuleList(self.resnet.children())[:-1]
            self.resnet = nn.Sequential(*features).append(nn.Flatten())

    def forward(self, x):
        # 应用ResNet模型直到平均池化层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # 在每个残差块后应用Simam_module
        x = self.resnet.layer1(x)
        x = self.simam(x)
        x = self.resnet.layer2(x)
        x = self.simam(x)
        x = self.resnet.layer3(x)
        x = self.simam(x)
        x = self.resnet.layer4(x)
        x = self.simam(x)

        # 应用平均池化和全连接层
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x