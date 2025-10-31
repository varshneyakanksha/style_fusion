from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import shutil, os, io, traceback
from style_transfer import image_loader, run_style_transfer
import torch

app = FastAPI(title="Style Fusion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMG_SIZE = 256
@app.post("/style-transfer")
async def style_transfer_endpoint(content: UploadFile = File(...), style: UploadFile = File(...), steps: int = 50):
    try:
        
        content_path = os.path.join(UPLOAD_DIR, "content.jpg")
        style_path   = os.path.join(UPLOAD_DIR, "style.jpg")
        with open(content_path, "wb") as f:
            shutil.copyfileobj(content.file, f)
        with open(style_path, "wb") as f:
            shutil.copyfileobj(style.file, f)

        
        content_tensor = image_loader(content_path, imsize=IMG_SIZE)
        style_tensor   = image_loader(style_path,   imsize=IMG_SIZE)

        print(f"[DEBUG] Content tensor shape: {tuple(content_tensor.shape)}")
        print(f"[DEBUG] Style tensor shape:   {tuple(style_tensor.shape)}")


        if content_tensor.shape != style_tensor.shape:
          raise RuntimeError(f"Shape mismatch: content={tuple(content_tensor.shape)}, style={tuple(style_tensor.shape)}")



        
        output = run_style_transfer(content_tensor, style_tensor, num_steps=steps)

        
        if isinstance(output, torch.Tensor):
            from torchvision import transforms
            output = output.detach().cpu().squeeze(0)
            output = transforms.ToPILImage()(output)

        
        buf = io.BytesIO()
        output.save(buf, format="JPEG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
