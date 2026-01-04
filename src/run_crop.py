import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from src.run import find_image_path, load_and_process_csv, print_stats


def print_image_matching_stats(dataset_folder: str, ground_truth_path: str):
    """
    Tìm tất cả file ảnh trong dataset_folder, so khớp với image_id trong CSV,
    in ra số lượng và tỷ lệ khớp.
    """
    dataset_root = Path(dataset_folder).expanduser().resolve()
    # Tìm tất cả file ảnh
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(dataset_root.rglob(ext))
    image_files_names = [p.name for p in image_files]
    print(f"[INFO] Tổng số file ảnh tìm thấy: {len(image_files_names)}")
    # Đọc image_id từ CSV
    gt_csv_path = Path(ground_truth_path).expanduser().resolve()
    image_ids = set()
    with open(gt_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img_id = row.get("image_id", "").strip()
            if img_id:
                image_ids.add(img_id)
    print(f"[INFO] Tổng số image_id trong CSV: {len(image_ids)}")
    # So khớp theo tên file (giữ nguyên đuôi)
    matched = set(image_files_names) & image_ids
    print(
        f"[INFO] Số lượng image_id khớp với file ảnh: {len(matched)} / {len(image_ids)} ({len(matched) / max(1, len(image_ids)) * 100:.2f}%)"
    )
    print(
        f"[INFO] Số lượng file ảnh khớp với image_id: {len(matched)} / {len(image_files_names)} ({len(matched) / max(1, len(image_files_names)) * 100:.2f}%)"
    )


def main(
    dataset_folder: str,
    ground_truth_path: str,
    output_folder: str,
):
    print("\n" + "=" * 60)
    print("BẮT ĐẦU WORKFLOW CROP")
    print("=" * 60 + "\n")

    # Bước kiểm tra khớp ảnh và image_id
    print("[Kiểm tra khớp ảnh với image_id trong CSV]")
    print_image_matching_stats(dataset_folder, ground_truth_path)

    # Bước 1 & 2: Load và xử lý CSV (tận dụng code run.py)
    print("[Bước 1-2] Đọc và xử lý ground-truth CSV...")
    fixed_csv_path, processed_rows, stats = load_and_process_csv(
        ground_truth_path, dataset_folder
    )

    # Bước 3: In thống kê
    if stats:
        print_stats(stats)

    print("\n" + "=" * 60)
    print("KẾT THÚC WORKFLOW CROP")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGM Crop Workflow")
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Root folder chứa các ảnh"
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Đường dẫn file CSV ground-truth",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Thư mục lưu kết quả"
    )

    args = parser.parse_args()

    main(
        dataset_folder=args.dataset_folder,
        ground_truth_path=args.ground_truth_path,
        output_folder=args.output_folder,
    )
