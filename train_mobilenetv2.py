import os
import datetime

import torch
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from torchsummary import summary


def create_model(num_classes):
    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512

    backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280  # 设置对应backbone输出特征通道数

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],  # 在哪些特征层进行roi
        output_size=[7, 7],  # roi_pool输出特征图大小
        sampling_ratio=2  # 采样率
    )
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training".format(device.type))

    # 用来保存coco_info的文件
    result_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 检查保存权重的文件是否存在，不存在就创建
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(prob=0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }




if __name__ == "__main__":
    net = create_model(1000)
    net.to("cuda:0")
    print(net)
