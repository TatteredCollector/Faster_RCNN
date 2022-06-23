import os
import random


def main():
    random.seed(0)  # 设置随机数种子，保证每次随机结果一致
    file_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(file_path), "path {} not exist!".format(file_path)

    val_rate = 0.5  # 划分比例

    file_name = sorted([file.split(".")[0] for file in os.listdir(file_path)])
    file_num = len(file_name)
    val_index = random.sample(range(0, file_num),
                              k=int(file_num * val_rate))  # 从序列seq中选择n个随机且独立的元素；
    train_files = []
    val_files = []
    for index, file_name in enumerate(file_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    try:
        train_f = open("train.txt", 'x')
        eval_f = open("val.txt", 'x')
        #  join()：连接字符串数组。将字符串、元组、
        #  列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
