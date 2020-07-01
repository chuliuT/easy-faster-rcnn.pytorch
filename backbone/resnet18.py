from typing import Tuple

import torchvision
from torch import nn

import backbone.base

# 定义一个 ResNet18 类，继承自backbone.base 里面的Base类
class ResNet18(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)#子类调用父类里面的初始化函数  赋值pretrained属性

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        #这里的调用PyTorch官方的模型，_pretrained的值是 上面 super 的pretrained的值初始化的值
        resnet18 = torchvision.models.resnet18(pretrained=self._pretrained)

        # list(resnet18.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear

        # 将modle里面的 层 转换为一个 list 后面会调用Sequential来构造 特征提取层
        children = list(resnet18.children())
        features = children[:-3]# 这里 取到【0-6】
        num_features_out = 256  #用于RPN的第一个卷积

        hidden = children[-3]#【7】# RCNN里面的隐藏层
        num_hidden_out = 512 # RCNN 分类的 Linear 输入层的 大小

        # 冷冻部分层的参数，不参与训练，【0-4】层，浅层这些抽取的结构基本功能都一样。
        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False#梯度属性设置为否，不需要计算梯度

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out
