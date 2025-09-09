import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os

# ----------- Config -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Paths - change if needed
content_path = "data/content/content.jpg"
style_path   = "data/styles/style_vangogh.jpg"
style_path   = "data/styles/style_madhubani.jpg"
output_path  = "outputs/output_stylized.jpg"

# Image size (smaller if CPU)
imsize = 512 if torch.cuda.is_available() else 256

# Weights
style_weight = 1e6
content_weight = 1e0
num_steps = 200  # reduce if slow

# ----------- Helpers -------------
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

unloader = transforms.Compose([
    transforms.Normalize(mean=[0.,0.,0.],
                         std=[1/0.229,1/0.224,1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1.,1.,1.]),
    transforms.ToPILImage()
])

def image_loader(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)  # add batch dimension
    return image.to(device, torch.float)

def save_image(tensor, path):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    image = unloader(tensor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print("Saved:", path)

# Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# ----------- Load images -------------
content_img = image_loader(content_path)
style_img = image_loader(style_path)

if content_img.size() != style_img.size():
    print("Resizing style image to match content size.")
    from torchvision import transforms as T
    resize = T.Resize(content_img.shape[-2:])
    style_img = resize(style_img)

# ----------- Load pretrained VGG and extract features -------------
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

style_layers_idx = [0, 5, 10, 19, 28]   # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
content_layer_idx = 21                  # conv4_2

def get_features(x, model, layers):
    features = {}
    for layer_idx, layer in enumerate(model):
        x = layer(x)
        if layer_idx in layers:
            features[layer_idx] = x
    return features

# Get content and style targets
with torch.no_grad():
    content_features = get_features(content_img, vgg, [content_layer_idx])
    style_features = get_features(style_img, vgg, style_layers_idx)
    style_grams = {lt: gram_matrix(style_features[lt]) for lt in style_features}

# ----------- Input image to optimize -------------
input_img = content_img.clone().requires_grad_(True).to(device)

# ----------- Optimizer -------------
optimizer = optim.LBFGS([input_img])

# ----------- Optimization loop -------------
print("Starting optimization...")
run = [0]
while run[0] <= num_steps:
    def closure():
        optimizer.zero_grad()
        input_content_feat = get_features(input_img, vgg, [content_layer_idx])
        input_style_feat = get_features(input_img, vgg, style_layers_idx)

        content_loss = content_weight * torch.nn.functional.mse_loss(
            input_content_feat[content_layer_idx], content_features[content_layer_idx]
        )

        style_loss = 0
        for lt in style_layers_idx:
            input_gram = gram_matrix(input_style_feat[lt])
            target_gram = style_grams[lt]
            style_loss += torch.nn.functional.mse_loss(input_gram, target_gram)
        style_loss *= style_weight

        loss = content_loss + style_loss
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0 or run[0] == 1 or run[0] == num_steps:
            print(f"Step {run[0]}: Style Loss {style_loss.item():.4f}, Content Loss {content_loss.item():.4f}")
        return loss

    optimizer.step(closure)

print("Optimization finished.")
save_image(input_img, output_path)
