# notebooks/02_multi_style_fusion.py
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Content & multiple styles
content_path = "data/content/content.jpg"


STYLE_PATHS = [
    "data/styles/style_vangogh.jpg",
    "data/styles/style_watercolor.jpg",
]

STYLE_WEIGHTS = [0.6, 0.4]

# Output
output_path  = "outputs/fusion_vangogh_watercolor_60_40.jpg"

# Speed/quality knobs
imsize      = 512 if torch.cuda.is_available() else 256
num_steps   = 200 if torch.cuda.is_available() else 150
style_weight   = 1e6
content_weight = 1.0

# ----------------- Helpers -----------------
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

def load_image(path):
    img = Image.open(path).convert("RGB")
    return loader(img).unsqueeze(0).to(device, torch.float)

def save_image(tensor, path):
    img = tensor.detach().cpu().clone().squeeze(0)
    img = unloader(img)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print("Saved:", path)

def gram_matrix(t):
    b, c, h, w = t.size()
    f = t.view(c, h*w)
    G = torch.mm(f, f.t())
    return G / (c*h*w)

# ----------------- Load images -----------------
content_img = load_image(content_path)
style_imgs  = [load_image(p) for p in STYLE_PATHS]

# Resize styles to content size (if needed)
from torchvision.transforms import functional as F
for i, s in enumerate(style_imgs):
    if s.shape[-2:] != content_img.shape[-2:]:
        style_imgs[i] = F.resize(s, content_img.shape[-2:])

# ----------------- VGG backbone -----------------
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
style_layers_idx   = [0, 5, 10, 19, 28]  # conv1_1 ... conv5_1
content_layer_idx  = 21                  # conv4_2

def get_features(x, model, layers):
    feats = {}
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layers:
            feats[i] = x
    return feats

# Targets
with torch.no_grad():
    content_target = get_features(content_img, vgg, [content_layer_idx])[content_layer_idx]

    # Per-style features & grams
    per_style_feats = [get_features(s, vgg, style_layers_idx) for s in style_imgs]
    per_style_grams = []
    for feats in per_style_feats:
        per_style_grams.append({lt: gram_matrix(feats[lt]) for lt in style_layers_idx})

    
    # Normalize weights to sum=1 for safety
    wsum = sum(STYLE_WEIGHTS)
    norm_w = [w/wsum for w in STYLE_WEIGHTS]
    fused_target_grams = {}
    for lt in style_layers_idx:
        fused = None
        for w, grams in zip(norm_w, per_style_grams):
            g = grams[lt]
            fused = (w*g) if fused is None else (fused + w*g)
        fused_target_grams[lt] = fused

# ----------------- Optimize pixels -----------------
input_img = content_img.clone().requires_grad_(True)
optimizer = optim.LBFGS([input_img])

print("Starting optimization...")
run = [0]
while run[0] <= num_steps:
    def closure():
        optimizer.zero_grad()
        feats_c = get_features(input_img, vgg, [content_layer_idx])[content_layer_idx]
        feats_s = get_features(input_img, vgg, style_layers_idx)

        # Content loss
        c_loss = content_weight * torch.nn.functional.mse_loss(feats_c, content_target)

        # Style loss w.r.t. fused targets
        s_loss = 0.0
        for lt in style_layers_idx:
            g_in = gram_matrix(feats_s[lt])
            g_tg = fused_target_grams[lt]
            s_loss = s_loss + torch.nn.functional.mse_loss(g_in, g_tg)
        s_loss = s_loss * style_weight

        loss = c_loss + s_loss
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0 or run[0] == 1 or run[0] == num_steps:
            print(f"Step {run[0]} | Style {s_loss.item():.4f} | Content {c_loss.item():.4f}")
        return loss
    optimizer.step(closure)

print("Optimization finished.")
save_image(input_img, output_path)
