#  call()的本质是将一个类变成一个函数
#  （使这个类的实例可以像函数一样调用）
#  一个类实例要变成一个可调用对象，只需要实现一个特殊方法__call__()。
# 允许一个类的实例像函数一样被调用。实质上说，这意味着 x() 与 x.__call__() 是相同的。注意 __call__ 参数可变。这意味着你可以定义 __call__ 为其他你想要的函数，无论有多少个参数。
# __call__ 在那些类的实例经常改变状态的时候会非常有效。
# 调用这个实例是一种改变这个对象状态的直接和优雅的做法。

import random
from torchvision.transforms import functional as F


class Compose(object):
    """transform多个的组合"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """PIL转化为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        # 标签已经在自定义数据类中转化
        return image, target


class RandomHorizontalFlip(object):
    # 水平翻转，x坐标会出现变化，
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # bbox xmin,ymin xmax,ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target
