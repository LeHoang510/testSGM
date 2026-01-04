from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
import numpy as np


def draw_bbox_on_image(
    img: Image.Image,
    bbox: Optional[Tuple[float, float, float, float]],
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    """
    Vẽ bounding box lên ảnh.
    bbox: (x, y, width, height)
    """
    if bbox is None:
        return img

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x, y, w, h = bbox

    # Clip bbox to image size
    img_w, img_h = img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return img_copy
