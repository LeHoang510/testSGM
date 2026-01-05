import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from src.pre_process import SGM_preprocess
from src.utils import find_image_path, print_stats
from src.csv_processor import load_and_process_csv
from PIL import Image


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


def crop_and_update_csv(
    dataset_folder: str,
    fixed_csv_path: Path,
    output_folder: str,
    model_path: str,
):
    """
    Crop ảnh theo bbox bằng SGM_preprocess, sau đó cập nhật lại các cột image_width, image_height, x, y, width, height, bbxs.
    Lưu ảnh crop vào output_folder, lưu metadata.csv đã update vào output_folder.
    """
    # Thực hiện crop bằng SGM_preprocess (YOLO), nhận về crop_info_dict
    cropped_folder, crop_info_dict = SGM_preprocess(
        input_root=dataset_folder,
        model_path=model_path,
        output_root=output_folder,
        overwrite=True,
        verbose=True,
    )

    # Đọc lại metadata_fixed.csv để cập nhật bbox cho metadata mới
    dataset_root = Path(dataset_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    updated_rows = []
    with open(fixed_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []

        image_path_key = None
        link_key = None
        for k in fieldnames:
            k_lower = k.lower()
            if k_lower == "image_path":
                image_path_key = k
            elif k_lower == "link":
                link_key = k

        for row in rdr:
            img_path_str = (
                row.get(image_path_key, "") if image_path_key else row.get(link_key, "")
            )
            img_path_str = img_path_str.strip().replace("\\", "/")
            if not img_path_str:
                updated_rows.append(row)
                continue

            out_img_path = output_root / img_path_str
            if not out_img_path.exists():
                print(f"[WARN] Không tìm thấy ảnh crop: {out_img_path}")
                updated_rows.append(row)
                continue

            try:
                # Get crop bbox info (x1, y1, x2, y2) from original image
                crop_bbox = crop_info_dict.get(img_path_str)
                if crop_bbox is None:
                    print(f"[WARN] Không tìm thấy crop info cho: {img_path_str}")
                    updated_rows.append(row)
                    continue

                crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox

                # Load cropped image to get new dimensions
                img = Image.open(out_img_path)
                img = img.convert("RGB")
                crop_w, crop_h = img.size

                # Get original lesion bbox from metadata
                try:
                    orig_x = float(row.get("x", 0))
                    orig_y = float(row.get("y", 0))
                    orig_width = float(row.get("width", 0))
                    orig_height = float(row.get("height", 0))
                except Exception:
                    # If bbox info is invalid, assume full cropped image
                    orig_x, orig_y, orig_width, orig_height = 0, 0, crop_w, crop_h

                # Translate bbox coordinates: subtract crop offset
                new_x = orig_x - crop_x1
                new_y = orig_y - crop_y1
                new_width = orig_width
                new_height = orig_height

                # Clip bbox to valid region within cropped image
                new_x = max(0, min(new_x, crop_w - 1))
                new_y = max(0, min(new_y, crop_h - 1))

                # Adjust width/height if bbox extends beyond image
                if new_x + new_width > crop_w:
                    new_width = crop_w - new_x
                if new_y + new_height > crop_h:
                    new_height = crop_h - new_y

                # Ensure minimum size
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                # Update row with new bbox info
                row["image_width"] = str(crop_w)
                row["image_height"] = str(crop_h)
                row["x"] = str(int(new_x))
                row["y"] = str(int(new_y))
                row["width"] = str(int(new_width))
                row["height"] = str(int(new_height))
                row["bbxs"] = str(
                    [(int(new_x), int(new_y), int(new_width), int(new_height))]
                )

                # Update image_path/link to new location (relative to output_folder)
                if image_path_key:
                    row[image_path_key] = str(out_img_path.relative_to(output_root))
                elif link_key:
                    row[link_key] = str(out_img_path.relative_to(output_root))

                updated_rows.append(row)
            except Exception as e:
                print(f"[ERROR] Update failed: {out_img_path} - {e}")
                updated_rows.append(row)

    # Lưu metadata.csv đã update vào output_folder
    out_fieldnames = fieldnames
    out_csv_path = output_root / "metadata.csv"
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"[INFO] Đã lưu metadata.csv đã update: {out_csv_path}")


def main(
    dataset_folder: str,
    ground_truth_path: str,
    output_folder: str,
    model_path: str,
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

    # Bước 4: Crop ảnh theo bbox và cập nhật metadata
    print("[Bước 4] Crop ảnh theo bbox và cập nhật metadata...")
    crop_and_update_csv(
        dataset_folder=dataset_folder,
        fixed_csv_path=fixed_csv_path,
        output_folder=output_folder,
        model_path=model_path,  # <-- truyền model_path vào
    )

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
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Đường dẫn model YOLO dùng để crop",
    )

    args = parser.parse_args()

    main(
        dataset_folder=args.dataset_folder,
        ground_truth_path=args.ground_truth_path,
        output_folder=args.output_folder,
        model_path=args.model_path,
    )
