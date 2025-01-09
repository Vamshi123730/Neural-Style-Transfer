import streamlit as st
from PIL import Image
from nst_tfhub import blend_styles
import os

# Create directories
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("output_images", exist_ok=True)

st.title("Neural Style Transfer with Two Styles")
st.write("Upload your content image and two style images to generate a masterpiece!")

# File upload widgets
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file_1 = st.file_uploader("Upload Style Image 1", type=["jpg", "png", "jpeg"])
style_file_2 = st.file_uploader("Upload Style Image 2", type=["jpg", "png", "jpeg"])

if content_file and style_file_1 and style_file_2:
    content_path = os.path.join("uploaded_images", "content_image.jpg")
    style_path_1 = os.path.join("uploaded_images", "style_image_1.jpg")
    style_path_2 = os.path.join("uploaded_images", "style_image_2.jpg")
    output_path = os.path.join("output_images", "blended_styled_image.jpg")

    # Save uploaded images
    with open(content_path, "wb") as f:
        f.write(content_file.getbuffer())
    with open(style_path_1, "wb") as f:
        f.write(style_file_1.getbuffer())
    with open(style_path_2, "wb") as f:
        f.write(style_file_2.getbuffer())

    # Display uploaded images
    content_image = Image.open(content_path)
    style_image_1 = Image.open(style_path_1)
    style_image_2 = Image.open(style_path_2)

    st.image([content_image, style_image_1, style_image_2], 
             caption=["Content Image", "Style Image 1", "Style Image 2"], 
             width=300)

    # Style blending weights
    weight_1 = st.slider("Weight for Style Image 1", 0.0, 1.0, 0.5)
    weight_2 = 1.0 - weight_1

    if st.button("Generate"):
        st.write("Blending styles and creating your masterpiece... This may take a few moments!")
        with st.spinner("Processing..."):
            blend_styles(content_path, style_path_1, style_path_2, output_path, weight_1, weight_2)
        st.success("Stylization complete!")
        stylized_image = Image.open(output_path)
        st.image(stylized_image, caption="Stylized Image", width=400)
