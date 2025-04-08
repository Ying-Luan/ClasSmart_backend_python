import torch
from torchvision.models import resnet50


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 8, kernel_size=5)
        self.gelu = torch.nn.GELU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(8 * 109 * 93, n_classes)

    def forward(self, x):
        x = self.pool(self.gelu(self.conv1(x)))  # (3, 448, 384) -> (32, 444, 380) -> (32, 222, 190)
        x = self.pool(self.gelu(self.conv2(x)))  # (32, 222, 190) -> (8, 218, 186) -> (8, 109, 93)
        # print(x.shape)
        x = x.view(-1, 8 * 109 * 93)
        x = self.fc(x)
        return x
    

class HugeCNN(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(64 *26 * 26, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (3, 224, 224) -> (32, 222, 222) -> (32, 111, 111)
        x = self.pool(self.relu(self.conv2(x)))  # (32, 111, 111) -> (64, 109, 109) -> (64, 54, 54)
        x = self.pool(self.relu(self.conv3(x)))  # (64, 54, 54) -> (64, 52, 52) -> (64, 26, 26)
        x = x.view(-1, 64 * 26 * 26)
        x = self.fc(x)

        return x
    

class RCNN(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = resnet50(pretrained=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
