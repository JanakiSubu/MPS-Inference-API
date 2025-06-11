import torch
from torchvision import transforms
from PIL import Image
import io

# Preprocess image bytes to tensor
async def preprocess(file, device, size=224):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    return tf(img).unsqueeze(0).to(device)