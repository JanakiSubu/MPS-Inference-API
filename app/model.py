import os
import torch
from torchvision import models

MODEL_TYPE = os.getenv('MODEL_TYPE', 'resnet')  # 'resnet' or 'vit'
QUANTIZE = os.getenv('QUANTIZE', '0') == '1'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load and optionally quantize model
def load_model():
    if MODEL_TYPE == 'vit':
        model = models.vit_b_16(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
    model.eval()
    if QUANTIZE:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    model.to(DEVICE)
    return model

# Single inference call
@torch.no_grad()
def infer(model, inputs):
    outputs = model(inputs)
    return outputs.argmax(dim=1)