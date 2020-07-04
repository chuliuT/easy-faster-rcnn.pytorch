import ast
from typing import Tuple, List

from roi.pooler import Pooler# Pooler类 有两个方法  ROI pooling ROI Align

# ref  https://www.cnblogs.com/wayne-tou/p/11896706.html
# https://www.runoob.com/python/python-func-callable.html
# https://blog.csdn.net/nanhuaibeian/article/details/102143356
# config的父类
class Config(object):
    #图像缩放的尺寸 短边缩放为600 长边为 1000
    IMAGE_MIN_SIDE: float = 600.0
    IMAGE_MAX_SIDE: float = 1000.0
    # Anchor box的比例 faster-rcnn原文  [0.5,1,2.0] 这里写成元组的形式
    ANCHOR_RATIOS: List[Tuple[int, int]] = [(1, 2), (1, 1), (2, 1)]
    ANCHOR_SIZES: List[int] = [128, 256, 512]
    POOLER_MODE: Pooler.Mode = Pooler.Mode.ALIGN # ROI pool的模式

    @classmethod#python中cls代表的是类的本身
    #cfg=Config()  # 应为注释了pool类所以这里 打印少了一个 pool—mode的值
    #print(cfg.describe())
    # Config:
    # ANCHOR_RATIOS = [(1, 2), (1, 1), (2, 1)]
    # ANCHOR_SIZES = [128, 256, 512]
    # IMAGE_MAX_SIDE = 1000.0
    # IMAGE_MIN_SIDE = 600.0
    def describe(cls):
        text = '\nConfig:\n'
        #callable 用于检查是否可用 getattr获取对象的属性
        attrs = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith('__')]
        text += '\n'.join(['\t{:s} = {:s}'.format(attr, str(getattr(cls, attr))) for attr in attrs]) + '\n'

        return text
    # 配置函数 可以自定义配置  ast literal_eval()函数：则会判断需要计算的内容计算后是不是合法的python类型，如果是则进行运算，否则就不进行运算
    #IMAGE_MIN_SIDE 短边
    #IMAGE_MAX_SIDE 长边
    #ANCHOR_RATIOS  anchor的比例
    #ANCHOR_SIZES   anchor的大小
    #POOLER_MODE    roi pooling align
    @classmethod
    def setup(cls, image_min_side: float = None, image_max_side: float = None,
              anchor_ratios: List[Tuple[int, int]] = None, anchor_sizes: List[int] = None, pooler_mode: str = None):
        if image_min_side is not None:
            cls.IMAGE_MIN_SIDE = image_min_side
        if image_max_side is not None:
            cls.IMAGE_MAX_SIDE = image_max_side

        if anchor_ratios is not None:
            cls.ANCHOR_RATIOS = ast.literal_eval(anchor_ratios)
        if anchor_sizes is not None:
            cls.ANCHOR_SIZES = ast.literal_eval(anchor_sizes)
        if pooler_mode is not None:
            # pass
            cls.POOLER_MODE = Pooler.Mode(pooler_mode)

if __name__ == '__main__':
    #调试时注释掉 请相关的pool类
    cfg=Config()
    print(cfg.describe())