import csv
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils import file_hash, find_image_path, filter_large_bboxes


def load_and_process_csv(
    ground_truth_path: str, dataset_folder: str, ready_csv: bool = False
) -> Tuple[Path, List[Dict], Dict]:
    """
    Bước 1: Đọc CSV, group theo image_path/link, gộp bbx, tạo cột cancer từ Classification
    Bước 2: Tạo cột split = "test", lưu metadata_fixed.csv
    Returns: (fixed_csv_path, processed_rows, stats)
    """
    gt_csv_path = Path(ground_truth_path).expanduser().resolve()
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {gt_csv_path}")

    if gt_csv_path.stem.endswith("_fixed"):
        print(f"[INFO] File đã được xử lý: {gt_csv_path}")
        return gt_csv_path, [], {}

    grouped_rows = {}
    dataset_folder_p = Path(dataset_folder).expanduser().resolve()

    with open(gt_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []

        image_path_key = None
        link_key = None
        cancer_key = None
        classification_key = None

        for k in fieldnames:
            k_lower = k.lower()
            if k_lower == "image_path":
                image_path_key = k
            elif k_lower == "link":
                link_key = k
            elif k_lower == "cancer":
                cancer_key = k
            elif k_lower == "classification":
                classification_key = k

        for row in rdr:
            group_key = None
            if image_path_key:
                group_key = row.get(image_path_key, "").strip()
            elif link_key:
                group_key = row.get(link_key, "").strip()

            if not group_key:
                continue

            group_key = group_key.replace("\\", "/")
            image_id = row.get("image_id", "").strip()
            need_update = False
            if image_path_key:
                img_path_val = row.get(image_path_key, "").strip().replace("\\", "/")
                if not img_path_val or not (dataset_folder_p / img_path_val).exists():
                    need_update = True
            elif link_key:
                img_path_val = row.get(link_key, "").strip().replace("\\", "/")
                if not img_path_val or not (dataset_folder_p / img_path_val).exists():
                    need_update = True
            else:
                img_path_val = ""
                need_update = True
            if need_update and image_id:
                found_path = find_image_path(dataset_folder_p, image_id)
                if found_path:
                    if image_path_key:
                        row[image_path_key] = found_path
                        group_key = found_path
                    elif link_key:
                        row[link_key] = found_path
                        group_key = found_path

            if group_key not in grouped_rows:
                grouped_rows[group_key] = {"row": row.copy(), "bbxs": []}

            if all(k in row for k in ["x", "y", "width", "height"]):
                try:
                    bbx = (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["width"]),
                        float(row["height"]),
                    )
                    grouped_rows[group_key]["bbxs"].append(bbx)
                except Exception:
                    pass

    processed_rows = []
    for group_key, info in grouped_rows.items():
        row = info["row"].copy()

        img_width = float(row.get("image_width", 0))
        img_height = float(row.get("image_height", 0))
        filtered_bbxs = filter_large_bboxes(
            info["bbxs"], img_width, img_height, threshold=0.9
        )

        row["bbxs"] = str(filtered_bbxs)
        row["split"] = "test"

        if not ready_csv:
            if not cancer_key and classification_key:
                classification_val = row.get(classification_key, "").strip().lower()
                if classification_val == "normal":
                    row["cancer"] = "0"
                elif classification_val in ["benign", "malignant"]:
                    row["cancer"] = "1"
                else:
                    row["cancer"] = ""

        processed_rows.append(row)

    fixed_csv_path = gt_csv_path.parent / f"{gt_csv_path.stem}_fixed.csv"
    need_save = True
    if fixed_csv_path.exists():
        try:
            if file_hash(gt_csv_path) == file_hash(fixed_csv_path):
                print(f"[INFO] File fixed giống file gốc, không cần lưu lại.")
                need_save = False
        except Exception:
            pass

    if need_save:
        out_fieldnames = list(fieldnames)
        if "bbxs" not in out_fieldnames:
            out_fieldnames.append("bbxs")
        if "split" not in out_fieldnames:
            out_fieldnames.append("split")
        if "cancer" not in out_fieldnames:
            out_fieldnames.append("cancer")

        with open(fixed_csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(processed_rows)

        print(f"[INFO] Đã lưu file ground-truth đã xử lý: {fixed_csv_path}")

    total_images = len(processed_rows)
    positive_count = sum(1 for r in processed_rows if r.get("cancer") == "1")
    negative_count = sum(1 for r in processed_rows if r.get("cancer") == "0")

    stats = {
        "total_images": total_images,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": positive_count / total_images if total_images > 0 else 0,
        "negative_ratio": negative_count / total_images if total_images > 0 else 0,
    }

    return fixed_csv_path, processed_rows, stats
