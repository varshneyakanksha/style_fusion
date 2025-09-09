import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy



# Image pre-processing
imsize = 512
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_file, imsize=512):
    image = Image.open(image_file).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # force same size for all images
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)



# Load VGG19 model
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x
    
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size, b=channels, (c,d)=feature map size
    features = input.view(a * b, c * d)  # reshape
    G = torch.mm(features, features.t())  # Gram matrix
    return G.div(a * b * c * d)  # normalize


# Style loss
class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_features).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input



# Build model with losses
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    model = nn.Sequential()
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

        if name == "conv_4":  # stop early
            break

    return model, style_losses, content_losses


# Run style transfer
def run_style_transfer(content_img, style_img, num_steps=200, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    input_img = content_img.clone()

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("Starting training...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            return style_score + content_score
        optimizer.step(closure)

    output = input_img.detach().cpu().clone().squeeze(0)
    output = unloader(output)
    return output
