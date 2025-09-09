import streamlit as st
from PIL import Image

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