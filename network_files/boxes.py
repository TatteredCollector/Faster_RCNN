import torch
from typing import Tuple
from torch import Tensor
import torchvision


def remove_small_boxes(boxes, min_size):
    # type:(Tensor,float)->Tensor
    """
        Remove boxes which contains at least one side smaller than min_size.
        移除宽高小于指定阈值的索引
        Arguments:
            boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
            min_size (float): minimum size

        Returns:
            keep (Tensor[K]): indices of the boxes that have both sides
                larger than min_size
    """
    # 预测boxes 的宽高
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # keep = (ws >= min_size) & (hs >= min_size)  # 当满足宽，高都大于给定阈值时为True
    # torch.ge(input,other)逐元素比较input和other，即是否 \( input >= other \)。
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    # type:(Tensor,tuple[int,int])->Tensor
    """
    Clip boxes so that they lie inside an image of size `size`.
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1,x2
    boxes_y = boxes[..., 1::2]  # y1,y2

    height, width = size
    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        #    clamp（）函数的功能将输入input张量每个元素的
        #    值压缩到区间 [min,max]，并返回结果到一个新张量。
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def nms(boxes, scores, iou_threshold):
    # type:(Tensor,Tensor,float)->Tensor
    """
    这里调用的官方底层函数，算法主要思想：
    1.计算最高score与其他框的IOU，
    2.高于iou_threshold的剔除，
    3.再次对保留下来除去最高score的进行排序，重复步骤1、2，
    直到所有框都进行了筛选
    :param boxes:
    :param scores:
    :param iou_threshold:
    :return: keep
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor,Tensor,Tensor,float)->Tensor
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # torch.numel()用来统计tensor中元素的个数
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()
    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
        Return intersection-over-union (Jaccard index) of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Arguments:
            boxes1 (Tensor[N, 4])
            boxes2 (Tensor[M, 4])

        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # 取左顶点的最大值
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    # 取右顶点最小值
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
