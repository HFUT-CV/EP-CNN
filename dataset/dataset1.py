# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from cutout import Cutout


class Hand(data.Dataset):
    r"""
    继承data.Dataset类，创建dataset对象，作为参数添加入DataLoader()类
    """
    def __init__(self, root, transforms=None, train=True, img_size=224):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root
        self.img_size = img_size
        self._read_txt_file()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 数据增强
            if not train:
                self.transforms = T.Compose([
                    T.Resize((self.img_size, self.img_size)),
                    # T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([           # 对图像进行各种转换操作，并用函数compose将这些转换操作组合起来；
                    T.Resize((self.img_size, self.img_size)),
                    # T.RandomSizedCrop(224),将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
                    T.RandomHorizontalFlip(),   # 以给定的概率随机水平旋转给定的图像，默认为0.5
                    T.ToTensor(),               # 将图像转化Tensor，即张量
                    normalize,                   # 归一化处理
                    # T.RandomErasing(),
                    Cutout(1, 28),
                ])

    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.images_root
        else:
            txt_file = self.images_root

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_path[index]
        # img_path = self.images_root + self.images_path[index]
        # print(img_path)
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')    # gray to rgb
        data = self.transforms(data)
        return data, int(label), img_path

    def __len__(self):
        return len(self.images_path)


if __name__ == '__main__':
    for i in range(20):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transforms = T.Compose([
            T.Resize((224, 224)),
            # T.CenterCrop(224),
            T.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            T.ToPILImage()
        ])

        test_img = Image.open('./2.jpg')
        test_img = transforms(test_img)
        test_img.save('./1_%d_transform.jpg' % i)