from typing import Tuple, Type
from torch import nn

#定义一个Backbone的父类，继承自object类，所有类的基类
class Base(object):
    #类的属性值
    OPTIONS = ['resnet18', 'resnet50', 'resnet101']
    # 静态方法，不强制传参，可以用 Base().func()  也可以用 Base.func()
    @staticmethod
    def from_name(name: str) -> Type['Base']:
        #name: str 指明name的类型是 str， -> 指定返回的类型是 Base
        if name == 'resnet18':#通过name来返回 不同的 backbone 如resnet18...
            from backbone.resnet18 import ResNet18# 子类的实现在resnet18中实现
            return ResNet18
        elif name == 'resnet50':
            from backbone.resnet50 import ResNet50
            return ResNet50
        elif name == 'resnet101':
            from backbone.resnet101 import ResNet101
            return ResNet101
        else:#name 是其他的str，则抛出一个值错误
            raise ValueError

    def __init__(self, pretrained: bool):#pretrained 的值类型是bool 布尔型
        super().__init__()
        # 初始化类内的属性（公有） 这里只有一个下划线
        self._pretrained = pretrained
    # 定义了一个函数 但是没有实现它，用于子类的 复写。输出做了类型检查 返回是一个tuple 
    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError
