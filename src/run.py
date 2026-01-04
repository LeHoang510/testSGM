import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from src.processing import (
    load_full_model,
    safe_convert_to_rgb,
    build_preprocess,
    predict_binary_label_and_confidence,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def load_and_process_csv(
    ground_truth_path: str, dataset_folder: str
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

            if group_key not in grouped_rows:
                grouped_rows[group_key] = {"row": row.copy(), "bbxs": []}

            # Collect bounding box nếu có
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

    # Xử lý và tạo metadata_fixed.csv
    processed_rows = []
    for group_key, info in grouped_rows.items():
        row = info["row"].copy()

        # Thêm cột bbxs
        row["bbxs"] = str(info["bbxs"])

        # Thêm cột split
        row["split"] = "test"

        # Tạo cột cancer từ Classification nếu cần
        if not cancer_key and classification_key:
            classification_val = row.get(classification_key, "").strip().lower()
            if classification_val == "normal":
                row["cancer"] = "0"
            elif classification_val in ["benign", "malignant"]:
                row["cancer"] = "1"
            else:
                row["cancer"] = ""

        processed_rows.append(row)

    # Lưu file metadata_fixed.csv
    fixed_csv_path = gt_csv_path.parent / f"{gt_csv_path.stem}_fixed.csv"
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

    # Tính stats
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


def print_stats(stats: Dict):
    """Bước 3: In thống kê về dataset"""
    print("\n" + "=" * 60)
    print("THỐNG KÊ DATASET")
    print("=" * 60)
    print(f"Tổng số ảnh: {stats['total_images']}")
    print(f"Positive: {stats['positive_count']} ({stats['positive_ratio'] * 100:.2f}%)")
    print(f"Negative: {stats['negative_count']} ({stats['negative_ratio'] * 100:.2f}%)")
    print("=" * 60 + "\n")


def run_predictions(
    dataset_folder: str,
    fixed_csv_path: Path,
    pretrained_model_path: str,
    output_folder: str,
    device: str = None,
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
    preprocess = build_preprocess(tuple(input_size), normalize)

    # Đọc fixed CSV
    image_list = []
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
            img_path_str = img_path_str.strip()
            if img_path_str:
                image_list.append((img_path_str, row))

                # Lấy ground truth label
                cancer_val = row.get("cancer", "").strip()
                try:
                    gt_label = int(float(cancer_val))
                    gt_map[img_path_str] = gt_label
                except Exception:
                    pass

    # Chạy predictions
    predictions_map = {}
    output_rows = []

    for img_path_str, row in tqdm(image_list, desc="Predictions", unit="img"):
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

    # Lưu metadata với predictions
    out_fieldnames = list(fieldnames) if fieldnames else []
    if "predict" not in out_fieldnames:
        out_fieldnames.append("predict")
    if "confidence" not in out_fieldnames:
        out_fieldnames.append("confidence")

    output_csv_path = output_root / "metadata_predictions.csv"
    with open(output_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[INFO] Đã lưu kết quả dự đoán: {output_csv_path}")

    return predictions_map, output_rows


def evaluate_predictions(predictions_map: Dict, gt_map: Dict):
    """Bước 5: Đánh giá kết quả predictions"""
    all_preds = []
    all_labels = []
    all_probs = []

    for img_path, (pred_label, conf) in predictions_map.items():
        if img_path in gt_map:
            gt_label = gt_map[img_path]
            all_preds.append(pred_label)
            all_labels.append(gt_label)
            all_probs.append(float(pred_label))

    if len(all_preds) == 0:
        print("[ERROR] Không có dữ liệu để đánh giá")
        return

    # Tính các chỉ số
    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    acc = correct / total

    try:
        auc = (
            roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
        )
    except Exception:
        auc = None

    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        precision = 0.0

    try:
        recall = recall_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        recall = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sen = recall
        spec = None

    # In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    acc_str = f"Accuracy: {acc * 100:.2f}%"
    if auc is not None:
        acc_str += f" | AUC: {auc * 100:.2f}%"
    print(acc_str)

    metrics_str = (
        f"Precision: {precision * 100:.2f}% | Sensitivity: {recall * 100:.2f}%"
    )
    if spec is not None:
        metrics_str += f" | Specificity: {spec * 100:.2f}%"
    print(metrics_str)
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(cm)
    print("=" * 60 + "\n")


def main(
    dataset_folder: str,
    ground_truth_path: str,
    pretrained_model_path: str,
    output_folder: str,
    device: str = None,
):
    """Main workflow"""
    print("\n" + "=" * 60)
    print("BẮT ĐẦU WORKFLOW")
    print("=" * 60 + "\n")

    # Bước 1 & 2: Load và xử lý CSV
    print("[Bước 1-2] Đọc và xử lý ground-truth CSV...")
    fixed_csv_path, processed_rows, stats = load_and_process_csv(
        ground_truth_path, dataset_folder
    )

    # Bước 3: In thống kê
    if stats:
        print_stats(stats)

    # Load lại từ fixed CSV để lấy gt_map
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
            img_path_str = img_path_str.strip()
            cancer_val = row.get("cancer", "").strip()
            try:
                gt_label = int(float(cancer_val))
                gt_map[img_path_str] = gt_label
            except Exception:
                pass

    # Bước 4: Chạy predictions
    print("[Bước 4] Chạy model predictions...")
    predictions_map, output_rows = run_predictions(
        dataset_folder=dataset_folder,
        fixed_csv_path=fixed_csv_path,
        pretrained_model_path=pretrained_model_path,
        output_folder=output_folder,
        device=device,
    )

    # Bước 5: Đánh giá
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

    args = parser.parse_args()

    main(
        dataset_folder=args.dataset_folder,
        ground_truth_path=args.ground_truth_path,
        pretrained_model_path=args.pretrained_model_path,
        output_folder=args.output_folder,
        device=args.device,
    )
