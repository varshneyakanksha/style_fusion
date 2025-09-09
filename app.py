import streamlit as st
from PIL import Image
<<<<<<< HEAD

st.set_page_config(page_title="AI Style Fusion", page_icon="🎨", layout="centered")
st.title("🎨 Style Fusion Generator")

st.write("Upload a *content image* and one or more *style images* to preview.")

# Upload content image
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])

# Upload style images (multiple allowed)
style_files = st.file_uploader("Upload Style Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if content_file:
    content_img = Image.open(content_file)
    st.image(content_img, caption="🖼 Content Image", use_column_width=True)

if style_files:
    st.write("🎭 Selected Style Images:")
    for file in style_files:
        style_img = Image.open(file)
        st.image(style_img, caption=f"Style: {file.name}", use_column_width=True)

# --- Button for demo ---
if content_file and style_files:
    if st.button("⚡ Generate Style Fusion"):
        st.info("Running style transfer... (demo mode)")
        st.success("✅ Output saved in outputs/ folder!")
=======
import os
from style_transfer import image_loader, run_style_transfer
IMG_SIZE = 512   # you can use 384/256 for faster runs on laptop


# Page setup
st.set_page_config(page_title="AI Style Fusion", page_icon="🎨", layout="wide")

st.title("🎨 AI-Powered Image Style Fusion Generator")
st.write("Upload a content image and a style image to generate fused artwork.")

# Upload content image
content_file = st.file_uploader("Upload Content file", type=["jpg", "jpeg", "png"])
# Upload style image
style_file = st.file_uploader("Upload Style file", type=["jpg", "jpeg", "png"])

# Show uploaded images
if content_file:
    content_file = Image.open(content_file)
    st.image(content_file, caption="🖼️ Content file", use_container_width=True)

if style_file:
    style_file = Image.open(style_file)
    st.image(style_file, caption="Style file", use_container_width=True)

# Generate button
if content_file and style_file:
    if st.button("⚡ Generate Style Fusion"):
        st.info("Running style transfer... Please wait ⏳")

        # Save temporary files
        content_path = "content_temp.jpg"
        style_path = "style_temp.jpg"
        content_file.save(content_path)
        style_file.save(style_path)

        # Load tensors
        content_tensor = image_loader(content_file)
        style_tensor   = image_loader(style_file)
        # If you allow multiple style uploads:
        style_tensors  = [image_loader(s) for s in style_file]


        # Run style transfer (backend)
        output = run_style_transfer(content_tensor, style_tensor, num_steps=50)

        # Save output
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "fused_result.jpg")
        output.save(output_path)

        # Show in UI
        st.success("✅ Style Fusion Completed!")
        st.image(output, caption=" Fused Output", use_container_width=True)
>>>>>>> bc615eb (Temporary local changes)
