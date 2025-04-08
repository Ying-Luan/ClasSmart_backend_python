import torch
import torch.nn.functional as F
from utils import *
from model import *


dimensions = get_image_dimension(directory='./data0')
eval_dimensions(dimensions=dimensions)
divide(directory="./data0", directory_target="./data", train_rate=0.9)

# 参数设置
learning_rate = 1e-4
epochs = 100
train_loader, test_loader = dataloader_generate()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 基础配置设置
# net = SimpleCNN().to(device)
net = RCNN().to(device)
# net = HugeCNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

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
    
    if epoch % 10 == 0 or epoch == epochs - 1:
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
        torch.save(net, "net.pth")
