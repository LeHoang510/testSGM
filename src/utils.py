import hashlib
from pathlib import Path
from typing import Optional, Tuple
import csv


def file_hash(filepath: Path) -> str:
    """Tính hash của file để so sánh"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image_path(dataset_folder: Path, image_id: str) -> str:
    """
    Tìm kiếm file ảnh thực tế trong dataset_folder dựa vào image_id.
    Ưu tiên: dataset_folder/patientID/image_id, nếu không có thì search toàn bộ subfolder.
    """
    patient_id = image_id.split("_")[0]
    candidate = dataset_folder / patient_id / image_id
    if candidate.exists():
        return str(candidate.relative_to(dataset_folder))
    for p in dataset_folder.rglob(image_id):
        try:
            return str(p.relative_to(dataset_folder))
        except Exception:
            continue
    return ""


def parse_bbox_from_row(row: dict) -> Optional[Tuple[float, float, float, float]]:
    """Parse bounding box từ row CSV (x, y, width, height)"""
    try:
        if all(k in row for k in ["x", "y", "width", "height"]):
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])
            return (x, y, w, h)
    except Exception:
        pass
    return None
