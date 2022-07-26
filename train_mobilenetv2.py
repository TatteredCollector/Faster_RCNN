import os
import datetime

import torch
import torchvision
from torch.utils import data

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

    VOC_root = "./"
    aspect_ratio_group_factor = 3
    batch_size = 8
    amp = False  # 是否使用混合精度训练 需要GPU支持volta架构以后的GPU，比如V100 A100

    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit does not in path: '{}'".format(VOC_root))

    # load train data set
    train_dataset = VOCDataSet(voc_root=VOC_root, year="2012", transforms=data_transform["train"], txt_name="train.txt")
    train_sampler = None

    # 是否按照图片相似高度比采样图片组成batch
    # 使用会减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        # 无放回地随机采样样本元素。
        train_sampler = data.RandomSampler(train_dataset)
        # 统计所有图像的宽高比例在bins 区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        # 每一个batch图片从同一宽高比例区间中选取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using %g dataloader workers" % nw)

    # 这里使用的collate_fn是自定义的，因为读取数据包括image和targets,不能直接使用默认的额
    if train_sampler:
        # 如果按照图片宽高比采样图片，dataloader中需要batch_sampler
        train_data_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )
    else:
        train_data_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )
    # load validation data set
    val_dataset = VOCDataSet(voc_root=VOC_root, year="2012", transforms=data_transform["val"], txt_name="val.txt")
    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    # create model num_class equal background + 20classea
    model = create_model(num_classes=21)

    model.to(device=device)

    scaler = torch.cuda.amp.GradScaler() if amp else None

    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                               #
    # 首先冻结前置backbone 训练rpn和预测网路                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    init_epochs = 5
    for epoch in range(init_epochs):
        # 训练一个epoch,每迭代10打印一次信息
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 验证集进行评价
        coco_info = utils.evaluate(model, val_data_loader, device)
        # write into txt
        with open(result_file, "a") as f:
            # 写入的数据包括coco指标 loss、learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, " ".join(result_info))
            f.write(txt + '\n')
        val_map.append(coco_info[1])  # pascal mAP

    torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                               #
    # 解冻前置backbone 训练网络权重                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 解冻backbone部分低层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # 重新定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)

    # 学习率调整
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    num_epochs = 20
    for epoch in range(init_epochs, num_epochs + init_epochs, 1):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 更新学习率
        lr_scheduler.step()

        coco_info = utils.evaluate(model, val_data_loader, device)

        #
        with open(result_file, 'a') as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # 仅保存最后5个epoch的权重
        if epoch in range(num_epochs + init_epochs)[-5:]:
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

    # 绘制损失函数和学习率曲线
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)
    # 绘制mAP
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    #main()
    import torchvision.models as models
    import torch
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        model = create_model(num_classes=21)
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
