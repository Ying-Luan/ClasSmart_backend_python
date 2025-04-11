import torch
from utils import *


# dimensions = get_image_dimension(directory='./data0')
# eval_dimensions(dimensions=dimensions)
# divide(directory="./data0", directory_target="./data", train_rate=0.9)

# 参数设置
model_typr = "OLD"
learning_rate = 1e-4
epochs = 100
train_loader, test_loader = dataloader_generate()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 8

# 基础配置设置
net = init_net(model_type=model_typr, n_classes=n_classes, device=device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

train(net=net,
      train_loader=train_loader,
      test_loader=test_loader,
      criterion=criterion,
      optimizer=optimizer,
      epochs=epochs,
      epoch_save=50,
      device=device)
