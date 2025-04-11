from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import json
import threading
from utils import train, dataloader_generate, unzip_data
import copy
import os


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

training_in_progress: bool = False
training_status: str = "idle"
training_progress: int = 0

updated_flag: bool = False

@app.post("/startTraining")
async def start_training(
    file: UploadFile = File(...),
    train_data_path: str = "",
    test_data_path: str = "",
    epochs: int = 20,
    learning_rate: float = 1e-4,
):
    """
    开始训练
    """
    global training_in_progress, training_status, training_progress, net, device

    if training_in_progress:
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    try:
        temp_dir = "temp_data"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        unzip_data(zip_path, temp_dir)
        pass

        train_loader, test_loader = dataloader_generate(train_data_path=temp_dir)

        net_training = copy.deepcopy(net).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        training_thread = threading.Thread(
            target=train,
            args=(net_training, train_loader, test_loader, criterion, optimizer, epochs, device),
        )
        training_thread.daemon = True
        training_thread.start()

        training_in_progress = True
        training_status = "running"
        training_progress = "running"

        return JSONResponse(content={"status": "Training started"})
    
    except Exception as e:
        tarining_status = "error"
        print(f"Error: {str(e)}")
        raise JSONResponse(content={"status": "Error starting training"})
    

def update_net():
    """
    更新网络模型
    """
    global net, device, updated_flag
    try:
        if updated_flag:
            new_net = torch.load("net.pth", weights_only=False, map_location=device).to(device)
            new_net.eval()
            net = new_net
            updated_flag = False
            return True
        else:
            return False
    except Exception as e:
        print(f"Error updating net: {str(e)}")
        return False


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        return JSONResponse(content={"type": "", "hugeType": hugeType, "error": 1})
    
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
    
# uvicorn main:app --host 0.0.0.0 --port 8000
