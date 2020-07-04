from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

# https://arxiv.org/pdf/1706.02677.pdf
class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1,
                 factor: float = 0.3333, num_iters: int = 500, last_epoch: int = -1):
        self.factor = factor
        self.num_iters = num_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self) -> List[float]:
        #学习率预热的过程
        if self.last_epoch < self.num_iters:#如果小于设置的阈值
            alpha = self.last_epoch / self.num_iters # 这里的alpha相当于 1-500 每次增大 1/500
            # factor因子 以 0.3333 为基础往上增长  最大是1
            factor = (1 - self.factor) * alpha + self.factor
        else:
            factor = 1
        #这里 原始的lr 是要乘以这个因子的
        return [lr * factor for lr in super().get_lr()]
