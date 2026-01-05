from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_predictions(predictions_map, gt_map):
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
