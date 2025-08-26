import streamlit as st
from PIL import Image
from io import BytesIO
import os

st.set_page_config(page_title="Style Fusion", page_icon="🎨", layout="centered")
st.title("AI-Powered Image Style Fusion Generator 🎨")
st.caption("Starter UI — hook your model later in style_fusion/*.py")

with st.expander("How to use"):
    st.markdown(
        "- Upload a **content image** (photo)\n"
        "- Upload one or more **style images** (art/texture)\n"
        "- Adjust **weights** and click **Generate**\n"
        "- Download the stylized result\n"
        "\nThis scaffold only wires the UI — add your model in `style_fusion/`.")
    
content_file = st.file_uploader("Upload Content Image", type=["jpg","jpeg","png"])
style_files  = st.file_uploader("Upload Style Images (1–3)", type=["jpg","jpeg","png"], accept_multiple_files=True)

w1 = st.slider("Style-1 weight", 0.0, 1.0, 0.7, 0.05)
w2 = st.slider("Style-2 weight", 0.0, 1.0, 0.3, 0.05)
w3 = st.slider("Style-3 weight (optional)", 0.0, 1.0, 0.0, 0.05)

if st.button("Generate"):
    if content_file is None or not style_files:
        st.error("Please upload a content image and at least one style image.")
    else:
        st.info("Model not yet connected — implement fusion in `style_fusion/` and call it here.")
        if content_file:
            st.image(Image.open(content_file), caption="Content", use_container_width=True)
        for i, f in enumerate(style_files, 1):
            st.image(Image.open(f), caption=f"Style {i}", use_container_width=True)
        st.write(f"Used weights: {w1}, {w2}, {w3}")
