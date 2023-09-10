import base64
from io import BytesIO
from PIL import Image
import random

def base642image(encoding):
    decoded_data = base64.b64decode(encoding)
    image_io = BytesIO(decoded_data)
    return Image.open(image_io).convert("RGB")
