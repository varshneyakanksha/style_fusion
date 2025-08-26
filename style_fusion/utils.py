from PIL import Image
import torch
from torchvision import transforms
import os

def load_image(path, size=512):
    img = Image.open(path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    return tfm(img).unsqueeze(0)  # [1,C,H,W]

def save_tensor_as_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tensor.clone().detach().cpu().squeeze(0).clamp(0,1)
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(path)
