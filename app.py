import streamlit as st
from PIL import Image
import os
from style_transfer import image_loader, run_style_transfer
import torch

# -------------------------------
# 🎨 Streamlit Web App
# -------------------------------
st.set_page_config(page_title="AI Style Fusion", page_icon="🎨", layout="centered")

st.title("🎨 AI-Powered Image Style Fusion Generator")
st.markdown("Upload a **content image** and a **style image** to generate artistic fusion!")

# -------------------------------
# 🖼️ Upload Section
# -------------------------------
content_file = st.file_uploader("📸 Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("🎭 Upload Style Image", type=["jpg", "jpeg", "png"])

# Preview uploaded images
col1, col2 = st.columns(2)
with col1:
    if content_file:
        content_image = Image.open(content_file)
        st.image(content_image, caption="Content Image", use_container_width=True)
with col2:
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, caption="Style Image", use_container_width=True)

# -------------------------------
# ⚙️ Parameters
# -------------------------------
st.sidebar.header("⚙️ Style Transfer Settings")
num_steps = st.sidebar.slider("Optimization Steps", 50, 500, 300, step=50)
style_weight = st.sidebar.slider("Style Strength", 1_000.0, 1_000_000.0, 100_000.0, step=10_000.0)

# -------------------------------
# 🚀 Generate Button
# -------------------------------
if st.button("⚡ Generate Style Fusion"):
    if content_file and style_file:
        with st.spinner("Running style transfer... Please wait ⏳"):
            try:
                # Save temporary files
                os.makedirs("temp", exist_ok=True)
                content_path = os.path.join("temp", "content.jpg")
                style_path = os.path.join("temp", "style.jpg")
                Image.open(content_file).save(content_path)
                Image.open(style_file).save(style_path)

                # Load tensors
                content_tensor = image_loader(content_path, imsize=256)
                style_tensor = image_loader(style_path, imsize=256)

                # Run Style Transfer
                output = run_style_transfer(content_tensor, style_tensor, num_steps=num_steps, style_weight=style_weight)

                # Display output
                st.success("✅ Style Fusion Completed Successfully!")
                st.image(output, caption="🖼️ Fused Artistic Output", use_container_width=True)

                # Save output
                os.makedirs("outputs", exist_ok=True)
                output.save("outputs/fused_result.jpg")
                st.download_button("⬇️ Download Result", data=open("outputs/fused_result.jpg", "rb").read(), file_name="fused_result.jpg")

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please upload both content and style images first!")
