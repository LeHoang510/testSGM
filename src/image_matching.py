import csv
from pathlib import Path


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
