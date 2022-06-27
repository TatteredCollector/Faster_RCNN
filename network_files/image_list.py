from typing import List, Tuple
from torch import Tensor


class ImageList(object):
    def __init__(self, tensors, image_size):
        # type: (Tensor,List[Tuple[int,int]])->None
        # -> 注解函数无返回值
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_size

    def to(self, device):
        # type:(Device)->ImageList
        cats_tensor = self.tensor.to(device)
        return ImageList(cats_tensor, self.image_sizes)
