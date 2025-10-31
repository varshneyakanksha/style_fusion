import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# --------------------------
# 🧠 Step 1: Basic Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 128

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

# --------------------------
# 🖼️ Step 2: Load and Preprocess Image
# --------------------------
def image_loader(image_file, imsize=256):
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.fit(image, (128, 128), method=Image.Resampling.BICUBIC)

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# --------------------------
# 🧩 Step 3: Normalization for VGG
# --------------------------
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# --------------------------
# 🧱 Step 4: Content + Style Loss
# --------------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        # Align matrix sizes
        min_size = min(G.size(0), self.target.size(0))
        G = G[:min_size, :min_size]
        target = self.target[:min_size, :min_size]
        self.loss = nn.functional.mse_loss(G, target)
        return input

# --------------------------
# 🏗️ Step 5: Build Style Model
# --------------------------
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name == "conv_4":
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss", content_loss)
            content_losses.append(content_loss)

        if name in ["conv_1", "conv_2", "conv_3", "conv_4"]:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss", style_loss)
            style_losses.append(style_loss)

        if name == "conv_4":
            break

    return model, style_losses, content_losses

# --------------------------
# ⚙️ Step 6: Run Style Transfer
# --------------------------
def run_style_transfer(content_img, style_img, num_steps=50, style_weight=1e5, content_weight=1):
    print("🚀 Fast style transfer started...")
    
    # CNN model and losses
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    
    input_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.02)  # 🔹 Adam = faster
    
    for step in range(num_steps):
        optimizer.zero_grad()
        model(input_img)
        
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        
        loss = style_score * style_weight + content_score * content_weight
        loss.backward(retain_graph=True)

        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}, Total Loss: {loss.item():.4f}")
    
    # Convert tensor back to image
    output = input_img.detach().cpu().clone().squeeze(0)
    output = unloader(output)
    return output
