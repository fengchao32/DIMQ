import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import datasets, transforms



# 路径设置
imagenet_root = '/ssddata/imagenet1k/ILSVRC2012'
train_dir = f'{imagenet_root}/train'
val_dir = f'{imagenet_root}/val'
train_label_file = f'{imagenet_root}/imagenet_100_train.txt'
val_label_file = f'{imagenet_root}/imagenet_100_eval.txt'

# 数据增强





train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	#进行归一化
])
#对测试集做变换
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

# train_dir = "/home/data/imagenet1k/ILSVRC2012/train"           #训练集路径
#train_dir = "D:/Dateset/Alldataset/mini-imagenet/train"
#train_dir = "D:/2021year/CVPR/PermuteNet-main/CNNonMNIST/data/trainNum_T/test"
#定义数据集
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
#加载数据集
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)#,num_workers=16,pin_memory=False

#val_dir = "D:/Dateset/Alldataset/mini-imagenet/val"
#val_dir = "D:/2021year/CVPR/PermuteNet-main/CNNonMNIST/data/trainNum_T/val"
# val_dir = "/home/data/imagenet1k/ILSVRC2012/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
test_loader = torch.utils.data.DataLoader(val_datasets, batch_size=256, shuffle=True,num_workers=8,pin_memory=True)




# 训练循环示例
# for images, labels in test_loader:
#     print(images.size(), labels.size())  # 打印当前 batch 的图片和标签尺寸
#     print(labels)  # 打印当前 batch 的标签
#     # 在这里进行训练步骤
