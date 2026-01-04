import os
import csv
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# def evaluate_model(
#     model, data_loader, device="cpu", mode="Test", return_loss=False, criterion=None
# ):
#     # Multi-GPU unwrap if needed
#     if isinstance(model, nn.DataParallel):
#         model = model.module
#     model = model.to(device)
#     model.eval()
#     correct = 0
#     total = 0
#     all_labels = []
#     all_preds = []
#     all_probs = []
#     total_loss = 0.0
#     # Use the same criterion as training if provided, else fallback to CrossEntropyLoss
#     if criterion is None:
#         criterion = nn.CrossEntropyLoss()

#     with torch.no_grad():
#         for batch in data_loader:
#             images, labels = batch
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             probs = torch.softmax(outputs, dim=1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#             all_probs.extend(
#                 probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy()
#             )

#     acc = correct / total
#     avg_loss = total_loss / total
#     # Tính các chỉ số
#     try:
#         auc = (
#             roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
#         )
#     except Exception:
#         auc = None
#     # Calculate precision and recall safely for binary/multiclass and degenerate cases
#     try:
#         precision = precision_score(
#             all_labels,
#             all_preds,
#             average="binary" if len(set(all_labels)) == 2 else "macro",
#             zero_division=0,
#         )
#     except Exception:
#         precision = 0.0
#         pass
#     try:
#         recall = recall_score(
#             all_labels,
#             all_preds,
#             average="binary" if len(set(all_labels)) == 2 else "macro",
#             zero_division=0,
#         )
#     except Exception:
#         recall = 0.0
#         pass
#     cm = confusion_matrix(all_labels, all_preds)
#     # Sensitivity (Recall) và Specificity
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#         sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#     else:
#         sen = recall
#         spec = None

#     # Gọn lại phần in ra
#     acc_loss_str = f"{mode} Accuracy : {acc * 100:.2f}% | Loss: {avg_loss:.4f}"
#     if auc is not None:
#         acc_loss_str += f" | AUC: {auc * 100:.2f}%"
#     print(acc_loss_str)
#     print(
#         f"{mode} Precision: {precision * 100:.2f}% | Sens: {recall * 100:.2f}%"
#         + (f" | Spec: {spec * 100:.2f}%" if spec is not None else "")
#     )

#     # In thông số tốt nhất nếu có
#     if hasattr(evaluate_model, "best_acc"):
#         best_str = f"Best Accuracy : {evaluate_model.best_acc * 100:.2f}%"
#         if hasattr(evaluate_model, "best_loss"):
#             best_str += f" | Loss: {evaluate_model.best_loss:.4f}"
#         if hasattr(evaluate_model, "best_auc") and evaluate_model.best_auc is not None:
#             best_str += f" | AUC: {evaluate_model.best_auc * 100:.2f}%"
#         print(best_str)
#     # Cập nhật best nếu tốt hơn
#     if not hasattr(evaluate_model, "best_acc") or acc > evaluate_model.best_acc:
#         evaluate_model.best_acc = acc
#         evaluate_model.best_loss = avg_loss
#         evaluate_model.best_auc = auc

#     print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

#     return (avg_loss, acc) if return_loss else acc


def evaluate_model(
    rows_by_folder: Optional[Dict] = None,
    ground_truth_csv: Optional[str] = None,
):
    """
    Evaluate predictions against ground truth.

    Args:
        rows_by_folder: Dict[Path, List[Dict]] from SGMpredict_folder containing predictions
                       Each dict has: "image_id", "label", "confidence_score"
        ground_truth_csv: Path to CSV with columns "image_id" and "cancer" (0 or 1)

    Computes and prints: accuracy, precision, recall, sensitivity, specificity, AUC,
    confusion matrix, and classification report.
    """

    if rows_by_folder is None or ground_truth_csv is None:
        print("[evaluate_model] rows_by_folder and ground_truth_csv required.")
        return {}

    # Flatten rows_by_folder into predictions dict
    preds_map: Dict[str, int] = {}  # image_id -> pred_label
    for rel_dir, rows in rows_by_folder.items():
        for r in rows:
            img_id = r.get("image_id", "").strip()
            if not img_id:
                continue
            try:
                lab = int(float(r.get("label", 0)))
            except Exception:
                lab = 0
            preds_map[img_id] = 1 if lab == 1 else 0

    # Load ground truth from CSV
    gt_csv_path = Path(ground_truth_csv).expanduser().resolve()
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {gt_csv_path}")

    gt_map: Dict[str, int] = {}  # image_id -> gt_label
    gt_rows: List[Dict] = []
    image_paths_set = set()
    image_ids_set = set()
    grouped_rows = {}

    with open(gt_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []
        has_split = "split" in fieldnames
        has_cancer = "cancer" in [k.lower() for k in fieldnames]
        abnormality_key = None
        for k in fieldnames:
            if k.lower() == "abnormality":
                abnormality_key = k
            if k.lower() == "image_path":
                image_path_key = k
            if k.lower() == "link":
                link_key = k
        # Group rows by image_path or link if present
        for row in rdr:
            # If split column missing, add it with value "test"
            if not has_split:
                row["split"] = "test"
            # If cancer column missing, create it from abnormality
            if not has_cancer and abnormality_key:
                ab_val = row.get(abnormality_key, "").strip().lower()
                if ab_val == "negative":
                    row["cancer"] = "0"
                elif ab_val == "positive":
                    row["cancer"] = "1"
                else:
                    row["cancer"] = ""
            # Determine grouping key
            group_key = None
            if "image_path_key" in locals():
                group_key = row.get(image_path_key, "").strip()
            elif "link_key" in locals():
                group_key = row.get(link_key, "").strip()
            else:
                group_key = row.get("image_id", "").strip()
            if not group_key:
                continue
            # Group bounding boxes
            if group_key not in grouped_rows:
                grouped_rows[group_key] = {"row": row, "bbxs": []}
            # Collect bounding box if present
            bbx = None
            if all(k in row for k in ["x", "y", "width", "height"]):
                try:
                    bbx = (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["width"]),
                        float(row["height"]),
                    )
                except Exception:
                    bbx = None
            if bbx:
                grouped_rows[group_key]["bbxs"].append(bbx)
        # Now process grouped_rows
        for group_key, info in grouped_rows.items():
            row = info["row"]
            img_id = row.get("image_id", "").strip()
            if img_id:
                image_ids_set.add(img_id)
            img_path = row.get("image_path", "").strip() or row.get("link", "").strip()
            if img_path:
                image_paths_set.add(img_path)
            cancer_raw = row.get("cancer", None)
            if cancer_raw is None or cancer_raw == "":
                continue
            try:
                cancer_lab = int(float(cancer_raw))
            except Exception:
                continue
            gt_map[img_id] = 1 if cancer_lab == 1 else 0
            # Optionally: you can store info["bbxs"] for later use if needed

    # Print statistics
    print(f"[STAT] Number of unique image_id: {len(image_ids_set)}")
    if image_paths_set:
        print(f"[STAT] Number of unique image_path: {len(image_paths_set)}")
    else:
        print(f"[STAT] No image_path column found in ground truth CSV.")

    # Align predictions with ground truth
    all_preds = []
    all_labels = []
    all_probs = []
    matched_count = 0

    for img_id, gt_lab in gt_map.items():
        if img_id in preds_map:
            pred_lab = preds_map[img_id]
            all_preds.append(pred_lab)
            all_labels.append(gt_lab)
            all_probs.append(float(pred_lab))
            matched_count += 1

    if matched_count == 0:
        print(
            "[LỖI] Không tìm thấy image_id trùng giữa dự đoán và file ground-truth CSV."
        )
        print(f"  Mẫu image_id từ dự đoán: {list(preds_map.keys())[:10]}")
        print(f"  Mẫu image_id từ ground-truth: {list(gt_map.keys())[:10]}")
        raise RuntimeError(
            "Không tìm thấy image_id trùng giữa dự đoán và ground-truth CSV."
        )

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total_loss = 0.0  # No loss computation for predictions, set to 0

    # Block not changed
    acc = correct / total
    avg_loss = total_loss / total
    # Tính các chỉ số
    try:
        auc = (
            roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
        )
    except Exception:
        auc = None
    # Calculate precision and recall safely for binary/multiclass and degenerate cases
    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        precision = 0.0
        pass
    try:
        recall = recall_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except Exception:
        recall = 0.0
        pass
    cm = confusion_matrix(all_labels, all_preds)
    # Sensitivity (Recall) và Specificity
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sen = recall
        spec = None

    # Gọn lại phần in ra
    mode = "Eval"
    acc_loss_str = f"{mode} Accuracy : {acc * 100:.2f}%"
    if auc is not None:
        acc_loss_str += f" | AUC: {auc * 100:.2f}%"
    print(acc_loss_str)
    print(
        f"{mode} Precision: {precision * 100:.2f}% | Sens: {recall * 100:.2f}%"
        + (f" | Spec: {spec * 100:.2f}%" if spec is not None else "")
    )

    # In thông số tốt nhất nếu có
    if hasattr(evaluate_model, "best_acc"):
        best_str = f"Best Accuracy : {evaluate_model.best_acc * 100:.2f}%"
        if hasattr(evaluate_model, "best_auc") and evaluate_model.best_auc is not None:
            best_str += f" | AUC: {evaluate_model.best_auc * 100:.2f}%"
        print(best_str)
    # Cập nhật best nếu tốt hơn
    if not hasattr(evaluate_model, "best_acc") or acc > evaluate_model.best_acc:
        evaluate_model.best_acc = acc
        evaluate_model.best_auc = auc

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    return acc
    # End Block not changed
