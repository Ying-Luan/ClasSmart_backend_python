from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import json
import threading
from utils import train, dataloader_generate, unzip_data, manage_folders, handle_images_match_label
import copy
import os
import shutil
    

### 相关参数定义
# 分类映射表
type2hugetpye = {"bottle": 0, "can": 0,  # 可回收物
                 "battery": 1, "smoke": 1,  # 有害垃圾
                 "fruit": 2, "vegetable": 2,  # 厨余垃圾
                 "brick": 3, "china": 3}  # 其他垃圾

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load("net.pth", weights_only=False, map_location=device)
net.eval()

min_confidence = 0.9  # 最小置信度

app = FastAPI()

transform = transforms.Compose([
        # transforms.Resize((448, 384)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])

with open("classdata.json", "r") as f:
    class2id = json.load(f)
id2class = {_id: _class for _class, _id in class2id.items()}

training_in_progress: bool = False  # 是否正在训练
updated_flag: bool = False  # 是否有可更新的网络模型


### 函数定义
@app.post("/startTraining")
async def start_training(
    file: UploadFile = File(...),
    count: int = 0,
) -> JSONResponse:
    """
    开始训练

    :param file: 训练数据 zip 文件
    :param count: 照片数量
    :return: 训练结果
    """

    if count <= 0:
        print("数据集为空，无法训练...")
        return JSONResponse(content={"code": 500})

    # 参数设置
    epochs: int = 100
    if count < 10:
        epochs *= 0.1
    elif count < 100:
        epochs *= 0.5
    elif count > 1000:
        epochs *= 2
    epoch_save: int = 50
    learning_rate: float = 1e-4
    global training_in_progress, net, device

    if training_in_progress:
        return JSONResponse(content={"code": 500})
    
    try:
        updated = update_net()
        
        # 整理数据
        temp_dir = "temp_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        unzip_data(zip_path, temp_dir, delete_zip_file=True)  # 解压并删除 zip 文件

        num_floders = manage_folders(main_folder=temp_dir, sub_folders=[floder for floder in class2id.keys()])
        
        if num_floders <= 0:
            print("没有可训练的数据集...")
            return JSONResponse(content={"code": 500})

        # TODO
        await handle_images_match_label(root_dir=temp_dir)

        train_loader, _ = dataloader_generate(data_train_root=temp_dir)

        def training_complete_callback():
            """
            训练完成回调函数

            :return:
            """
            global training_in_progress, updated_flag
            training_in_progress = False
            updated_flag = True
            shutil.rmtree(temp_dir)
            print("回调函数执行完毕...")

        # 开始训练
        net_training = copy.deepcopy(net).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        training_thread = threading.Thread(
            target=train,
            args=(net_training, train_loader, None, criterion, optimizer, int(epochs), epoch_save, device, training_complete_callback),
        )
        training_thread.daemon = True
        training_thread.start()

        training_in_progress = True

        return JSONResponse(content={"code": 200})
    
    except Exception as e:
        tarining_in_progress = False
        print(f"Error: {str(e)}")
        return JSONResponse(content={"code": 500})
    

def update_net() -> bool:
    """
    更新网络模型

    :return: 是否更新成功
    """
    global net, device, updated_flag
    try:
        if updated_flag:
            new_net = torch.load("net.pth", weights_only=False, map_location=device).to(device)
            new_net.eval()
            net = new_net
            updated_flag = False
            print("网络模型更新成功...")
            return True
        else:
            return False
    except Exception as e:
        print(f"Error updating net: {str(e)}")
        return False


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    预测垃圾类型

    :param file: 图片文件
    :return: 垃圾类型和大类
    """
    try:
        updated = update_net()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(image)

        output = torch.nn.functional.softmax(output, dim=1)
        max_prod, predicted_idx = torch.max(output, 1)
        print(f"max_prod: {max_prod.item():.4f}")
        predict_class = id2class[predicted_idx.item()]
        hugeType: int = type2hugetpye.get(predict_class, -1)
        if max_prod.item() < min_confidence:
            return JSONResponse(content={"type": predict_class, "hugeType": hugeType, "error": 2})

        return JSONResponse(content={"type": predict_class, "hugeType": hugeType, "error": 0})
    
    except Exception as e:
        return JSONResponse(content={"type": "", "hugeType": -1, "error": 1})
    
    
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
    
# uvicorn main:app --host 0.0.0.0 --port 8000
