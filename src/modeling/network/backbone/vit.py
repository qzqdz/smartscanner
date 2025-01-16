from torch import nn
from torchvision import models
from src.utils.registry import REGISTRY




@REGISTRY.register('vit')
class ViTModel(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(ViTModel, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)  # 加载预训练的ViT模型

        if classify:
            # 替换ViT的分类头
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        else:
            # 移除分类头，保留ViT作为特征提取器
            self.vit.heads = nn.Identity()

    def forward(self, inputs):
        return self.vit(inputs)

    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'head' in param_tuple[0], self.vit.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'head' not in param_tuple[0], self.vit.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers
        }
        return param_groups
