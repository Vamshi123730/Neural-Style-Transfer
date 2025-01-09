import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load TensorFlow Hub style transfer model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to preprocess an image
def load_image(image_path, max_dim=512):
    image = Image.open(image_path)
    image = image.convert('RGB')
    long_dim = max(image.size)
    scale = max_dim / long_dim
    new_size = tuple((np.array(image.size) * scale).astype(int))
    image = image.resize(new_size, Image.Resampling.LANCZOS)  # Updated from Image.ANTIALIAS
    image = np.array(image) / 255.0
    return tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

# Function to blend styles and apply them to the content image
def blend_styles(content_path, style_path_1, style_path_2, output_path, weight_1=0.5, weight_2=0.5):
    content_image = load_image(content_path)
    style_image_1 = load_image(style_path_1)
    style_image_2 = load_image(style_path_2)

    # Generate stylized images for both styles
    stylized_image_1 = model(content_image, style_image_1)[0]
    stylized_image_2 = model(content_image, style_image_2)[0]

    # Blend the two styles
    blended_stylized_image = weight_1 * stylized_image_1 + weight_2 * stylized_image_2

    # Save the blended stylized image
    blended_stylized_image = tf.squeeze(blended_stylized_image)
    blended_stylized_image = (blended_stylized_image.numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(blended_stylized_image)
    final_image.save(output_path)
