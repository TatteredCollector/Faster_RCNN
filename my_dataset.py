from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    def __init__(self, voc_root, year='2012', transforms=None, txt_name: str = 'train.txt'):
        assert year in ["2007", "2012"], "year must be in ['2007','2012']"
        # f’{}’ 用法等同于 format用法的简单使用，
        self.root = os.path.join(voc_root, 'VOCdevkit', "VOC"f"{year}")
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, "Annotations")

        # 读取 train.txt 或者val.txt
        txt_path = os.path.join(self.root, 'ImageSets', 'Main', txt_name)
        assert os.path.exists(txt_path), "not found {} ".format(txt_path)

        #  strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）
        with open(txt_path) as r:
            # r.readlines() 返回读取的每一行值+\n 列表 readline 就一行值
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in r.readlines() if len(line.strip()) > 0]

        # 检查所有xml 文件的可存在性
        # print(self.xml_list)
        assert len(self.xml_list) > 0, "in {} file does not find any information".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found {} file ".format(xml_path)

        json_file = "./pascal_voc_classes.json"
        assert os.path.exists(json_file), "{} file not exist".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transform = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # 读取xml文件内容
        xml_path = self.xml_list[idx]
        # print(xml_path)
        with open(xml_path, "r") as f:
            xml_str = f.read()
        # 法是将xml格式转化为Element 对象，Element 对象代表 XML 文档中的一个元素。
        # 元素可以包含属性、其他元素或文本。如果一个元素包含文本，则在文本节点中表示该文本。
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data["filename"])
        assert os.path.exists(img_path),"{} do not exist".format(img_path)
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("image {} format not JPEG".format(img_path))

        # 获取目标信息
        boxes = []
        labels = []
        iscrowd = []  # 目标之间是否重叠

        assert "object" in data, "{} lack of object information ".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            # 检查数据中是否有标注矩形框不符合规范
            if xmax <= xmin or ymax <= ymin:
                print("warning: in {} xml .there are some bbox w/h<=0".format(xml_path))
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # 转换成张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 构造返回值
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def get_height_and_width(self, idx):
        # 多GPU训练时候需要重新定义
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = data["size"]["height"]
        data_width = data["size"]["width"]
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        :param xml:
        Element对象，
        :return:xml转化为字典形式
        """
        if len(xml) == 0:  # 不在包含其他元素
            return {xml.tag: xml.text}
        result = {}

        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}  # 结果加上最外层

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
        idx: 输入需要获取图像的索引
        """
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        # 生成元组对（图像，目标）
        return tuple(zip(*batch))


# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
# #print(category_index)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=20):
#     img, target = train_data_set[index]
#     #print(type(img))
#     print(target["boxes"].numpy(),)
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
