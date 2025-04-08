from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import json


type2hugetpye = {"bottle": 0, "can": 0,  # 可回收物
                 "battery": 1, "smoke": 1,  # 有害垃圾
                 "fruit": 2, "vegetable": 2,  # 厨余垃圾
                 "brick": 3, "china": 3}  # 其他垃圾

app = FastAPI()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load("net.pth", weights_only=False, map_location=device)
net.eval()

transform = transforms.Compose([
        # transforms.Resize((448, 384)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, .225])
    ])

with open("classdata.json", "r") as f:
    class2id = json.load(f)
id2class = {_id: _class for _class, _id in class2id.items()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(image)

        _, predicted_idx = torch.max(output, 1)
        predict_class = id2class[predicted_idx.item()]
        hugeType: int = type2hugetpye.get(predict_class, -1)

        return JSONResponse(content={"type": predict_class, "hugeType": hugeType, "error": 0})
    
    except Exception as e:
        return JSONResponse(content={"type": "", "hugeType": hugeType, "error": 1})
    
# uvicorn main:app --host 0.0.0.0 --port 8000
