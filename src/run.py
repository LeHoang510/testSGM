import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import hashlib

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from src.processing import (
    load_full_model,
    safe_convert_to_rgb,
    build_preprocess,
    predict_binary_label_and_confidence,
    gradcam_heatmap_based,
    overlay_otsu,
)
from src.utils import file_hash, find_image_path, print_stats, filter_large_bboxes
from src.plot import parse_bbxs_from_string, draw_bboxes_on_image
from src.metrics import evaluate_predictions
from src.image_matching import print_image_matching_stats


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

    # Check if already fixed
    if gt_csv_path.stem.endswith("_fixed"):
        print(f"[INFO] File đã được xử lý: {gt_csv_path}")
        return gt_csv_path, [], {}

    grouped_rows = {}

    dataset_folder_p = Path(dataset_folder).expanduser().resolve()

    with open(gt_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []

        # Tìm các cột cần thiết
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

        # Group rows
        for row in rdr:
            # Xác định grouping key từ image_path hoặc link
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

        # Nếu đã có cột cancer và ready_csv=True thì giữ nguyên, không tạo lại
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


def run_predictions(
    dataset_folder: str,
    fixed_csv_path: Path,
    pretrained_model_path: str,
    output_folder: str,
    gt_map: Dict = None,  # Add gt_map parameter
    device: str = None,
    save_gradcam: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """
    Bước 4: Chạy model qua tất cả ảnh, lưu kết quả
    Returns: (predictions_map, output_rows)
    """
    dataset_root = Path(dataset_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Load model
    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Using device: {dev}")

    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        load_full_model(pretrained_model_path)
    )
    model = model.to(dev).eval()
    target_layer = gradcam_layer or "layer4"
    preprocess = build_preprocess(tuple(input_size), normalize)

    # Đọc fixed CSV
    image_list = []
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
            img_path_str = img_path_str.strip()
            img_path_str = img_path_str.replace("\\", "/")

            if img_path_str:
                image_list.append((img_path_str, row))

    # Group images by folder
    images_by_folder = defaultdict(list)
    for img_path_str, row in image_list:
        rel_path = Path(img_path_str)
        folder_key = rel_path.parent
        images_by_folder[folder_key].append((img_path_str, row))

    # Chạy predictions
    predictions_map = {}
    output_rows = []

    # Process each folder
    for folder_key in tqdm(
        sorted(images_by_folder.keys()), desc="Processing folders", unit="folder"
    ):
        folder_metadata = []

        # Remove inner tqdm for images, just use a normal for loop
        for img_path_str, row in images_by_folder[folder_key]:
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = dataset_root / img_path

            if not img_path.exists():
                print(f"[WARN] Không tìm thấy ảnh: {img_path}")
                continue

            # Load và predict
            img = safe_convert_to_rgb(Image.open(img_path))
            x = preprocess(img).unsqueeze(0).to(dev)

            with torch.no_grad():
                logits = model(x)

            label, conf, _ = predict_binary_label_and_confidence(logits)

            # Lưu kết quả
            predictions_map[img_path_str] = (label, conf)

            # Tạo output row
            out_row = row.copy()
            out_row["predict"] = str(label)
            out_row["confidence"] = f"{conf:.6f}"
            output_rows.append(out_row)

            # Get ground truth for this image
            gt_label = gt_map.get(img_path_str, "") if gt_map else ""

            # Add to folder metadata with ground truth
            folder_metadata.append(
                {
                    "image_id": img_path.name,
                    "ground_truth": str(gt_label) if gt_label != "" else "",
                    "label": str(label),
                    "confidence_score": f"{conf:.6f}",
                }
            )

            # Lưu ảnh và gradcam nếu cần
            if save_gradcam:
                img_out_dir = output_root / folder_key
                img_out_dir.mkdir(parents=True, exist_ok=True)

                # Parse bbxs và vẽ lên ảnh gốc
                bbxs_str = row.get("bbxs", "[]")
                bbxs = parse_bbxs_from_string(bbxs_str)
                if bbxs:
                    img_with_bbox = draw_bboxes_on_image(
                        img, bbxs, color="red", width=3
                    )
                else:
                    img_with_bbox = img

                # Lưu ảnh gốc (có bbox nếu có)
                img_out_path = img_out_dir / f"{img_path.stem}.png"
                try:
                    img_with_bbox.save(img_out_path, format="PNG")
                except Exception as e:
                    print(f"[ERROR] Không thể lưu ảnh gốc {img_path.stem}: {e}")

                # Tạo và lưu gradcam
                gradcam_out_path = img_out_dir / f"{img_path.stem}_gradcam.png"
                if label == 1:
                    try:
                        if arch_type is None or str(arch_type).lower() == "based":
                            cam = gradcam_heatmap_based(
                                model, x, target_layer, class_idx=1
                            )
                            overlay = overlay_otsu(img, cam, alpha=0.55)
                            overlay.save(gradcam_out_path, format="PNG")
                        else:
                            try:
                                from src.gradcam.gradcam_utils_patch import (
                                    pre_mil_gradcam,
                                    mil_gradcam,
                                )
                                import numpy as np

                                model_tuple = (
                                    model,
                                    tuple(input_size),
                                    model_name,
                                    target_layer,
                                    normalize,
                                    num_patches,
                                    arch_type,
                                )
                                (
                                    model_out,
                                    input_tensor,
                                    img0,
                                    layer0,
                                    class_idx0,
                                    pred_class0,
                                    prob0,
                                ) = pre_mil_gradcam(model_tuple, str(img_path))
                                cam = mil_gradcam(
                                    model_out, input_tensor, layer0, class_idx=1
                                )
                                if cam.ndim == 3:
                                    cam = cam.max(axis=0).astype(np.uint8)
                                overlay = overlay_otsu(img, cam, alpha=0.55)
                                overlay.save(gradcam_out_path, format="PNG")
                            except Exception as e_mil:
                                print(
                                    f"[WARN] MIL Gradcam failed for {img_path.stem}: {e_mil}, saving original"
                                )
                                img.save(gradcam_out_path, format="PNG")
                    except Exception as e:
                        print(
                            f"[WARN] Gradcam failed for {img_path.stem}: {e}, saving original"
                        )
                        try:
                            img.save(gradcam_out_path, format="PNG")
                        except Exception as e_save:
                            print(
                                f"[ERROR] Không thể lưu gradcam cho {img_path.stem}: {e_save}"
                            )
                else:
                    try:
                        img.save(gradcam_out_path, format="PNG")
                    except Exception as e:
                        print(
                            f"[ERROR] Không thể lưu gradcam (label=0) cho {img_path.stem}: {e}"
                        )

        # Save metadata.csv for this folder immediately after processing all images
        folder_csv_path = output_root / folder_key / "metadata.csv"
        folder_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(folder_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["image_id", "ground_truth", "label", "confidence_score"]
            )
            w.writeheader()
            w.writerows(folder_metadata)
        # print(f"[INFO] Đã lưu metadata.csv cho folder: {folder_key}")

    # Lưu metadata.csv tổng
    out_fieldnames = list(fieldnames) if fieldnames else []
    if "predict" not in out_fieldnames:
        out_fieldnames.append("predict")
    if "confidence" not in out_fieldnames:
        out_fieldnames.append("confidence")

    output_csv_path = output_root / "metadata.csv"
    with open(output_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[INFO] Đã lưu kết quả dự đoán tổng: {output_csv_path}")

    return predictions_map, output_rows


def main(
    dataset_folder: str,
    ground_truth_path: str,
    pretrained_model_path: str,
    output_folder: str,
    device: str = None,
    save_gradcam: bool = True,
    ready_csv: bool = False,
):
    """Main workflow"""
    print("\n" + "=" * 60)
    print("BẮT ĐẦU WORKFLOW")
    print("=" * 60 + "\n")

    print("[Kiểm tra khớp ảnh với image_id trong CSV]")
    print_image_matching_stats(dataset_folder, ground_truth_path)

    print("[Bước 1-2] Đọc và xử lý ground-truth CSV...")
    fixed_csv_path, processed_rows, stats = load_and_process_csv(
        ground_truth_path, dataset_folder, ready_csv=ready_csv
    )

    if stats:
        print_stats(stats)

    gt_map = {}
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
            cancer_val = row.get("cancer", "").strip()
            try:
                gt_label = int(float(cancer_val))
                gt_map[img_path_str] = gt_label
            except Exception:
                pass

    print("[Bước 4] Chạy model predictions...")
    predictions_map, output_rows = run_predictions(
        dataset_folder=dataset_folder,
        fixed_csv_path=fixed_csv_path,
        pretrained_model_path=pretrained_model_path,
        output_folder=output_folder,
        gt_map=gt_map,
        device=device,
        save_gradcam=save_gradcam,
    )

    print("[Bước 5] Đánh giá kết quả...")
    evaluate_predictions(predictions_map, gt_map)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH WORKFLOW")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGM Prediction Workflow")
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
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Đường dẫn model pretrained",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device để chạy model (cuda/cpu)"
    )
    parser.add_argument(
        "--save_gradcam", action="store_true", help="Lưu ảnh gradcam và ảnh gốc"
    )
    parser.add_argument(
        "--ready_csv",
        action="store_true",
        help="CSV đã có cột cancer, không cần tạo lại",
    )

    args = parser.parse_args()

    main(
        dataset_folder=args.dataset_folder,
        ground_truth_path=args.ground_truth_path,
        pretrained_model_path=args.pretrained_model_path,
        output_folder=args.output_folder,
        device=args.device,
        save_gradcam=args.save_gradcam,
        ready_csv=args.ready_csv,
    )
