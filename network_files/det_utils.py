import torch
import math
from typing import List, Tuple
from torch import Tensor


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 15)):
        # type: (Tuple[float,float,float,float],float)->None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type:(List[Tensor],List[Tensor])->List[Tensor]
        """
               结合anchors和与之对应的gt计算regression参数
               Args:
                   reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
                   proposals: List[Tensor] anchors/proposals

               Returns: regression parameters

         """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同

        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]  # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths  # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        # ::a 间隔a进行采样，保持原来变量的维度
        dx = rel_codes[:, 0::4] / wx  # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy  # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww  # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh  # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        # 将输入input张量每个元素的夹紧到区间[min, max]，
        # 并返回结果到一个新张量。
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pre_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pre_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pre_w = torch.exp(dw) * widths[:, None]
        pre_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pre_ctr_x - torch.tensor(0.5, dtype=pre_ctr_x.dtype, device=pre_w.device) * pre_w
        # ymin
        pred_boxes2 = pre_ctr_y - torch.tensor(0.5, dtype=pre_ctr_y.dtype, device=pre_h.device) * pre_h
        # xmax
        pred_boxes3 = pre_ctr_x + torch.tensor(0.5, dtype=pre_ctr_x.dtype, device=pre_w.device) * pre_w
        # ymax
        pred_boxes4 = pre_ctr_y + torch.tensor(0.5, dtype=pre_ctr_y.dtype, device=pre_h.device) * pre_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes

    def decode(self, rel_codes, boxes):
        # type:(Tensor,List[Tensor])-> Tensor
        """

        Args:
            rel_codes: bbox regression parameters
            boxes: anchors/proposals

        Returns:
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes
