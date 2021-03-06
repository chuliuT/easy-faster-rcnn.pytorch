import random
from enum import Enum
from typing import Tuple, List, Type, Iterator

import PIL
import torch.utils.data.dataset
import torch.utils.data.sampler
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import transforms

#数据集的基类，继承自torch的 Dataset类
class Base(torch.utils.data.dataset.Dataset):

    class Mode(Enum):#Mode为枚举类型
        TRAIN = 'train'#训练模式
        EVAL = 'eval'#验证模式
    #数据集的类型
    OPTIONS = ['voc2007', 'coco2017', 'voc2007-cat-dog', 'coco2017-person', 'coco2017-car', 'coco2017-animal']
    #一个静态方法，返回的类型为 Base类 的子类
    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'voc2007':
            #这里的dataset为 package的dataset  不要跟上面的 torch的dataset混淆了
            from dataset.voc2007 import VOC2007
            return VOC2007
        elif name == 'coco2017':
            from dataset.coco2017 import COCO2017
            return COCO2017
        elif name == 'voc2007-cat-dog':
            from dataset.voc2007_cat_dog import VOC2007CatDog
            return VOC2007CatDog
        elif name == 'coco2017-person':
            from dataset.coco2017_person import COCO2017Person
            return COCO2017Person
        elif name == 'coco2017-car':
            from dataset.coco2017_car import COCO2017Car
            return COCO2017Car
        elif name == 'coco2017-animal':
            from dataset.coco2017_animal import COCO2017Animal
            return COCO2017Animal
        else:
            raise ValueError

    def __init__(self, path_to_data_dir: str, mode: Mode, image_min_side: float, image_max_side: float):
        self._path_to_data_dir = path_to_data_dir# 数据集的路径
        self._mode = mode # 模式
        self._image_min_side = image_min_side#短边
        self._image_max_side = image_max_side#长边

    def __len__(self) -> int:#继承torch.utils.data.dataset.Dataset 必须实现的方法
        raise NotImplementedError
    # 定义一个获取 图像 image_id, image, scale, bboxes, labels 在子类里面实现
    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError
    # 计算 mAP 和 每个类的AP
    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        raise NotImplementedError
    # 将测试的结果  写入 txt 的文件里
    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        raise NotImplementedError

    @property
    def image_ratios(self) -> List[float]:# 返回图像的宽/高的比例
        raise NotImplementedError

    @staticmethod
    def num_classes() -> int:# 数据集的类别
        raise NotImplementedError

    # 用于图像的  短边和长边的处理  短边缩短为 image_min_side  长边：image_max_side
    @staticmethod
    def preprocess(image: PIL.Image.Image, image_min_side: float, image_max_side: float) -> Tuple[Tensor, float]:
        # resize according to the rules:
        #   1. scale shorter side to IMAGE_MIN_SIDE
        #   2. after scaling, if longer side > IMAGE_MAX_SIDE, scale longer side to IMAGE_MAX_SIDE
        scale_for_shorter_side = image_min_side / min(image.width, image.height)# 短边的尺度缩放因子  image_min_side：600
        longer_side_after_scaling = max(image.width, image.height) * scale_for_shorter_side #长边的缩放因子
        scale_for_longer_side = (image_max_side / longer_side_after_scaling) if longer_side_after_scaling > image_max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side

        transform = transforms.Compose([
            transforms.Resize((round(image.height * scale), round(image.width * scale))),  # interpolation `BILINEAR` is applied by default
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, scale

    @staticmethod
    def padding_collate_fn(batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch = zip(*batch)
        #找到batch里面最大的 width height bboxes长度 和labels长度
        max_image_width = max([it.shape[2] for it in image_batch])
        max_image_height = max([it.shape[1] for it in image_batch])
        max_bboxes_length = max([len(it) for it in bboxes_batch])
        max_labels_length = max([len(it) for it in labels_batch])
        #给这些图像做padding
        padded_image_batch = []
        padded_bboxes_batch = []
        padded_labels_batch = []
        #padding就是用batch里最大的width 和 height来计算 需要填充的距离 
        for image in image_batch:
            padded_image = F.pad(input=image, pad=(0, max_image_width - image.shape[2], 0, max_image_height - image.shape[1]))  # pad has format (left, right, top, bottom)
            padded_image_batch.append(padded_image)
        #
        for bboxes in bboxes_batch:
            padded_bboxes = torch.cat([bboxes, torch.zeros(max_bboxes_length - len(bboxes), 4).to(bboxes)])
            padded_bboxes_batch.append(padded_bboxes)

        for labels in labels_batch:
            padded_labels = torch.cat([labels, torch.zeros(max_labels_length - len(labels)).to(labels)])
            padded_labels_batch.append(padded_labels)

        image_id_batch = list(image_id_batch)
        #堆叠起来
        padded_image_batch = torch.stack(padded_image_batch, dim=0)
        scale_batch = torch.stack(scale_batch, dim=0)
        padded_bboxes_batch = torch.stack(padded_bboxes_batch, dim=0)
        padded_labels_batch = torch.stack(padded_labels_batch, dim=0)

        return image_id_batch, padded_image_batch, scale_batch, padded_bboxes_batch, padded_labels_batch
    #最近比例的随机采样
    class NearestRatioRandomSampler(torch.utils.data.sampler.Sampler):

        def __init__(self, image_ratios: List[float], num_neighbors: int):
            super().__init__(data_source=None)
            self._image_ratios = image_ratios#比例
            self._num_neighbors = num_neighbors#邻近的个数

        def __len__(self) -> int:
            return len(self._image_ratios)#返回长度

        def __iter__(self) -> Iterator[int]:
            image_ratios = torch.tensor(self._image_ratios)#变成tensor
            tall_indices = (image_ratios < 1).nonzero().view(-1)
            fat_indices = (image_ratios >= 1).nonzero().view(-1)

            tall_indices_length = len(tall_indices)#个数
            fat_indices_length = len(fat_indices)#个数

            tall_indices = tall_indices[torch.randperm(tall_indices_length)]
            fat_indices = fat_indices[torch.randperm(fat_indices_length)]

            num_tall_remainder = tall_indices_length % self._num_neighbors
            num_fat_remainder = fat_indices_length % self._num_neighbors

            tall_indices = tall_indices[:tall_indices_length - num_tall_remainder]
            fat_indices = fat_indices[:fat_indices_length - num_fat_remainder]

            tall_indices = tall_indices.view(-1, self._num_neighbors)
            fat_indices = fat_indices.view(-1, self._num_neighbors)
            merge_indices = torch.cat([tall_indices, fat_indices], dim=0)
            merge_indices = merge_indices[torch.randperm(len(merge_indices))].view(-1)

            return iter(merge_indices.tolist())
