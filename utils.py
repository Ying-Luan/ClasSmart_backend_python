import os
from PIL import Image
import random
import shutil
from typing import List, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import torch
from model import *
import zipfile


def get_image_dimension(directory: str):
    """
    获取图片尺寸数据
    
    :param directory: 图片目录
    :return:
    """
    dimensions = []
    for directory_low in os.listdir(directory):
        directory_full = os.path.join(directory, directory_low)
        if not os.path.isdir(directory_full):
            continue
        for filename in os.listdir(directory_full):
            filepath = os.path.join(directory_full, filename)
            with Image.open(filepath) as img:
                width, height = img.size
                dimensions.append((filepath, width, height))
    return dimensions


def eval_dimensions(dimensions: List[Tuple[str, int, int]]):
    """
    对图片数据进行统计处理

    :param dimensions: 图片尺寸数据
    :return:
    """
    widths = []
    heights = []
    for _, width, height in dimensions:
        widths.append(width)
        heights.append(height)
    print(f"Average width: {sum(widths) / len(widths):.2f}")
    print(f"Average height: {sum(heights) / len(heights):.2f}")


def divide(directory: str, directory_target: str, train_rate: float):
    """
    划分训练集和测试集

    :param directory: 原始数据集目录
    :param directory_target: 目标数据集目录
    :param train_rate: 训练集占比
    :return:
    """
    if os.path.exists(directory_target):
        shutil.rmtree(directory_target)
        os.makedirs(directory_target)

    assert 0 < train_rate < 1

    dir_train = os.path.join(directory_target, "train")
    dir_test = os.path.join(directory_target, "test")
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    for dir_low in os.listdir(directory):
        dir_full = os.path.join(directory, dir_low)
        if not os.path.isdir(dir_full):
            continue

        if dir_low in ["battery_1", "battery_5", "battery_7"]:
            target_dir_name = "battery"
            prefix = {
                "battery_1": "battery_1_",
                "battery_5": "battery_5_",
                "battery_7": "battery_7_",
            }[dir_low]
        else:
            target_dir_name = dir_low
            prefix = ""

        if dir_low == "zhuankuai":
            target_dir_name = "brick"

        jpgs = [os.path.join(dir_full, jpg) for jpg in os.listdir(dir_full)]

        random.shuffle(jpgs)
        split_index = int(len(jpgs) * train_rate)
        jpgs_train = jpgs[:split_index]
        jpgs_test = jpgs[split_index:]

        os.makedirs(os.path.join(dir_train, target_dir_name), exist_ok=True)
        os.makedirs(os.path.join(dir_test, target_dir_name), exist_ok=True)
        for jpg in jpgs_train:
            filename = os.path.basename(jpg)
            shutil.copy(jpg, os.path.join(dir_train, target_dir_name, prefix + filename))
        for jpg in jpgs_test:
            filename = os.path.basename(jpg)
            shutil.copy(jpg, os.path.join(dir_test, target_dir_name, prefix + filename))


def init_net(model_type: str, n_classes: int, device: torch.device):
    """
    初始化模型

    :param model_type: 模型类型
    :param n_classes: 类别数
    :param device: 设备
    :return:
    """
    assert model_type in ["RCNN", "HugeCNN", "SimpleCNN", "OLD"]
    if model_type == "RCNN":
        net = RCNN(n_classes=n_classes).to(device)
    elif model_type == "HugeCNN":
        net = HugeCNN(n_classes=n_classes).to(device)
    elif model_type == "SimpleCNN":
        net = SimpleCNN(n_classes=n_classes).to(device)
    elif model_type == "OLD":
        net = torch.load("net.pth", weights_only=False, map_location=device).to(device)
    
    return net


def dataloader_generate(data_train_root: str, data_test_root: Union[str, None] = None, class_to_idx: dict = None, json_save:bool = False):
    """
    生成训练集和测试集的 dataloader

    :param data_train_root: 训练集根目录
    :param data_test_root: 测试集根目录
    :param class_to_idx: 类别到索引的映射
    :return:
    """
    transform = transforms.Compose([
        # transforms.Resize((448, 384)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])

    train_dataset = datasets.ImageFolder(root=data_train_root, transform=transform)
    if data_test_root is not None:
        test_dataset = datasets.ImageFolder(root=data_test_root, transform=transform)

    if json_save:
        with open("classdata.json", "w") as f:
            json.dump(train_dataset.class_to_idx, f)
    
    if class_to_idx is not None:
        train_dataset.class_to_idx = class_to_idx
        if data_test_root is not None:
            test_dataset.class_to_idx = class_to_idx

    # print(train_dataset.class_to_idx)
    # print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    if data_test_root is not None:
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    else:
        test_loader = None

    return train_loader, test_loader


def train(net, train_loader, test_loader, criterion, optimizer, epochs: int, epoch_save: int, device: torch.device, training_complete_callback: callable = None):
    """
    训练模型

    :param net: 模型
    :param train_loader: 训练集 dataloader
    :param test_loader: 测试集 dataloader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    :param epoch_save: 保存模型的轮数
    :param device: 设备
    :param training_complete_callback: 训练完成回调函数
    :return:
    """
    net.train()
    print("开始训练...")
    for epoch in range(epochs):
        losses = []
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            y = net(x)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        print(f"epoch {epoch} loss: {sum(losses) / len(losses):.5f}")
    
        if test_loader is not None and (epoch % 10 == 0 or epoch == epochs - 1):
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    input, target = data
                    input, target = input.to(device), target.to(device)

                    y = net(input)
                    loss = criterion(y, target)
                    _, predicted = torch.max(y.data, dim=1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            print(f'Accuracy on test: {correct}/{total} = {correct / total * 100:.1f}%')

        # 保存模型
        if epoch % epoch_save == 0 or epoch == epochs - 1:
            torch.save(net, "net.pth")
            print("模型已保存...")
            
    if training_complete_callback is not None:
        training_complete_callback()
    print("训练结束...")


def unzip_data(zip_file_path: str, target_dir_path: str, delete_zip_file: bool = False):
    """
    解压数据集
    
    :param target_dir_path: 解压目标路径
    :param target_dir_path: 解压目标路径
    :param delete_zip_file: 是否删除压缩文件
    :return:
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir_path)
    if delete_zip_file:
        os.remove(zip_file_path)
    print(f"文件已解压到 {target_dir_path}")


def manage_folders(main_folder: str, sub_folders: List[str]):
    """
    管理文件夹

    :param main_folder: 主文件夹
    :param sub_folders: 子文件夹列表
    :return: 在 sub_folders 中存在的子文件夹数量
    """
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    # 当前文件夹下存在的子文件夹
    existing_folders = [floder for floder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, floder))]

    # 删除多余的子文件夹
    for floder in existing_folders:
        if floder not in sub_folders:
            shutil.rmtree(os.path.join(main_folder, floder))

    # for floder in sub_folders:
    #     if floder not in existing_folders:
    #         os.makedirs(os.path.join(main_folder, floder))

    return len([floder
                for floder in os.listdir(main_folder)
                if os.path.isdir(os.path.join(main_folder, floder))and floder in sub_folders])


if __name__ == "__main__":
    dimensions = get_image_dimension('./data0')
    eval_dimensions(dimensions)
    divide("./data0", "./data", 0.9)
    dataloader_generate()
