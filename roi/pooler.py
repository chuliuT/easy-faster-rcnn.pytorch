from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from support.layer.roi_align import ROIAlign

class Pooler(object):

    class Mode(Enum):#继承自枚举类
        POOLING = 'pooling'#roipooling
        ALIGN = 'align'#roialign

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, proposal_batch_indices: Tensor, mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16#图像下采样的倍数
        output_size = (7 * 2, 7 * 2)#14*14

        if mode == Pooler.Mode.POOLING:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                #先  取出建议框的 对应坐标  round四舍五入来得到一个 近似的featuremap上的位置
                start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                #再pooling变成 14*14    --》接下来再做一次pooling 变成 7*7
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                features,
                torch.cat([proposal_batch_indices.view(-1, 1).float(), proposal_bboxes], dim=1)
            )
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)#7*7
        return pool

