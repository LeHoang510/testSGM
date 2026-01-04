import ast
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw


def parse_bbxs_from_string(bbxs_str: str) -> List[Tuple[float, float, float, float]]:
    """Parse bbxs string to list of tuples"""
    try:
        bbxs = ast.literal_eval(bbxs_str)
        if isinstance(bbxs, list):
            return bbxs
    except Exception:
        pass
    return []


def draw_bboxes_on_image(
    img: Image.Image,
    bbxs: List[Tuple[float, float, float, float]],
    color="red",
    width=3,
) -> Image.Image:
    """Draw bounding boxes on image"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for bbox in bbxs:
        if len(bbox) == 4:
            x, y, w, h = bbox
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=width)

    return img_copy
