import hashlib
import csv
from pathlib import Path
from typing import Dict, List, Tuple


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


def print_stats(stats: Dict):
    """In thống kê về dataset"""
    print("\n" + "=" * 60)
    print("THỐNG KÊ DATASET")
    print("=" * 60)
    print(f"Tổng số ảnh: {stats['total_images']}")
    print(f"Positive: {stats['positive_count']} ({stats['positive_ratio'] * 100:.2f}%)")
    print(f"Negative: {stats['negative_count']} ({stats['negative_ratio'] * 100:.2f}%)")
    print("=" * 60 + "\n")


def print_image_matching_stats(dataset_folder: str, ground_truth_path: str):
    """
    Tìm tất cả file ảnh trong dataset_folder, so khớp với image_id trong CSV,
    in ra số lượng và tỷ lệ khớp.
    """
    dataset_root = Path(dataset_folder).expanduser().resolve()
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(dataset_root.rglob(ext))
    image_files_names = [p.name for p in image_files]
    print(f"[INFO] Tổng số file ảnh tìm thấy: {len(image_files_names)}")

    gt_csv_path = Path(ground_truth_path).expanduser().resolve()
    image_ids = set()
    with open(gt_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img_id = row.get("image_id", "").strip()
            if img_id:
                image_ids.add(img_id)
    print(f"[INFO] Tổng số image_id trong CSV: {len(image_ids)}")

    matched = set(image_files_names) & image_ids
    print(
        f"[INFO] Số lượng image_id khớp với file ảnh: {len(matched)} / {len(image_ids)} ({len(matched) / max(1, len(image_ids)) * 100:.2f}%)"
    )
    print(
        f"[INFO] Số lượng file ảnh khớp với image_id: {len(matched)} / {len(image_files_names)} ({len(matched) / max(1, len(image_files_names)) * 100:.2f}%)"
    )


def filter_large_bboxes(
    bbxs: List[Tuple[float, float, float, float]],
    img_width: float,
    img_height: float,
    threshold: float = 0.9,
) -> List[Tuple[float, float, float, float]]:
    """
    Lọc bỏ các bbox chiếm trên threshold (mặc định 90%) diện tích ảnh
    """
    if not bbxs:
        return bbxs

    img_area = img_width * img_height
    if img_area == 0:
        return bbxs

    filtered = []
    for bbox in bbxs:
        if len(bbox) == 4:
            x, y, w, h = bbox
            bbox_area = w * h
            ratio = bbox_area / img_area
            if ratio < threshold:
                filtered.append(bbox)

    return filtered
