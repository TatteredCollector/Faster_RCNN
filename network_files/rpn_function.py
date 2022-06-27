from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


class AnchorsGenerator(nn.Module):
    # 模块中有个特殊属性__annotations__
    # 用于收集模块中变量的注解，
    # 但这些注解同样也不会创建对应的变量。
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, list[torch.Tensor]]
    }
    """
        anchors生成器
        Module that generates anchors for a set of feature maps and
        image sizes.

        The module support computing anchors at multiple sizes and aspect ratios
        per feature map.

        sizes and aspect_ratios should have the same number of elements, and it should
        correspond to the number of feature maps.

        sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
        and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
        per spatial location for feature map i.

        Arguments:
            sizes (Tuple[Tuple[int]]):
            aspect_ratios (Tuple[Tuple[float]]):
        """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int],List[float],torch.dtype,torch.device)->Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios
        # [r1, r2, r3]' * [s1, s2, s3]
        # 长宽比例变化，在同一尺度下的面积不变
        # number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top, right-bottom coordinate relative to anchor center(0, 0)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round四舍五入

    def set_cell_anchors(self, dtype, device):
        # type:(torch.dtype,torch.device)->None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios, in zip(self.sizes, self.aspect_ratios)
        ]

        self.cell_anchors = cell_anchors

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type:(List[List[int]],List[List[Tensor]])->List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        # 获取anchors模板
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(行) 生成0-(grid_width-1)
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(列)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width] [y_shape,x_shape]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_y = shift_y.reshape(-1)
            shift_x = shift_x.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))
            return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int],List[List[Tensor]]])-> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        #  根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息
        # （这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchor_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchor_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
            # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
            # anchors是个list，每个元素为一张图像的所有anchors信息
            anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
            # Clear the cache in case that memory leaks.
            self._cache.clear()
            return anchors


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3滑动窗口
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels=in_channels, out_channels=num_anchors,
                                    kernel_size=1, stride=1, padding=0)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels=in_channels, out_channels=4 * num_anchors,
                                   kernel_size=1, stride=1, padding=0)
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type:(List[tensor])->Tuple[List[Tensor],List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    """
        Implements Region Proposal Network (RPN).

        Arguments:
            anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
                maps.
            head (nn.Module): module that computes the objectness and regression deltas
            fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
                considered as positive during training of the RPN.
            bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
                considered as negative during training of the RPN.
            batch_size_per_image (int): number of anchors that are sampled during training of the RPN
                for computing the loss
            positive_fraction (float): proportion of positive anchors in a mini-batch during training
                of the RPN
            pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
                contain two fields: training and testing, to allow for different values depending
                on training or evaluation
            post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
                contain two fields: training and testing, to allow for different values depending
                on training or evaluation
            nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

        """
