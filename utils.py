import os
from PIL import Image
import random
import shutil
from typing import List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json


def get_image_dimension(directory: str):
    """
    获取图片尺寸数据
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


def dataloader_generate():
    """
    生成训练集和测试集的 dataloader
    """
    transform = transforms.Compose([
        # transforms.Resize((448, 384)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, .225])
    ])

    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

    with open("classdata.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f)
    # print(train_dataset.class_to_idx)
    # print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    dimensions = get_image_dimension('./data0')
    eval_dimensions(dimensions)
    divide("./data0", "./data", 0.9)
    dataloader_generate()
