import torch

from torch import Tensor

#https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/smooth_l1_loss.py
# 相比与原来的smooth_L1 多了一个 beta参数
def beta_smooth_l1_loss(input: Tensor, target: Tensor, beta: float) -> Tensor:
    diff = torch.abs(input - target)# 计算 坐标的 差值
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = loss.sum() / (input.numel() + 1e-8)# 数值稳定 1e-8
    return loss
