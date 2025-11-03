import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ IMAGE LOADING ------------------
imsize = 256  # smaller = faster
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

def image_loader(image_file, imsize=256):
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.fit(image, (imsize, imsize), method=Image.Resampling.BICUBIC)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# ------------------ NORMALIZATION ------------------
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# ------------------ MODEL SETUP ------------------
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# ------------------ CONTENT LOSS ------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# ------------------ STYLE LOSS ------------------
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)

        # --- FIX: match smallest channel size ---
        min_c = min(G.size(0), self.target.size(0))
        G = G[:min_c, :min_c]
        target = self.target[:min_c, :min_c]

        self.loss = nn.functional.mse_loss(G, target)
        return input

# ------------------ BUILD MODEL ------------------
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

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
        else:
            continue

        model.add_module(name, layer)

        if name == "conv_4":
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in ["conv_1", "conv_2", "conv_3", "conv_4"]:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        if name == "conv_4":
            break

    return model, style_losses, content_losses

# ------------------ RUN STYLE TRANSFER ------------------
def run_style_transfer(content_img, style_img, num_steps=100, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

    input_img = content_img.clone()
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.02)

    print("✨ Running Style Transfer...")
    for step in range(num_steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()
        input_img.data.clamp_(0, 1)  # keep image valid

    output = input_img.detach().cpu().squeeze(0)
    return unloader(output)
