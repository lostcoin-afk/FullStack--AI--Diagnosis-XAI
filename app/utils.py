import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def decode_base64_to_image(base64_str, output_path):
    image_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as f:
        f.write(image_data)

def pil_image_to_base64(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
