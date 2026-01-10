import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback nếu không có tqdm

VIEW_ORDER = [("CC", "R"), ("CC", "L"), ("MLO", "R"), ("MLO", "L")]

GRADCAM_RE = re.compile(
    r"^(?P<base>.+?)_(?P<view>CC|MLO)_(?P<side>L|R)(?:_(?P<idx>\d+))?_gradcam\.png$",
    re.IGNORECASE,
)


def _read_result_csv(csv_path: Path) -> Dict[str, Tuple[int, float]]:
    m: Dict[str, Tuple[int, float]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = Path(row["image_id"].strip()).stem  # xxx_CC_R_2
            m[stem] = (int(row["label"]), float(row["confidence_score"]))
    return m


def _parse_gradcam_name(name: str) -> Optional[Tuple[str, str, str, Optional[int]]]:
    m = GRADCAM_RE.match(name)
    if not m:
        return None
    base = m.group("base")
    view = m.group("view").upper()
    side = m.group("side").upper()
    idx = m.group("idx")
    return base, view, side, (int(idx) if idx else None)


def _blank_image(size=(512, 512)) -> Image.Image:
    w, h = size
    return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 255)


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _compact_title(
    view: str, side: str, label: Optional[int], conf: Optional[float]
) -> str:
    token = f"{view}_{side}"
    if label is None or conf is None:
        return f"{token} | NA"
    status = "Positive" if label == 1 else "Negative"
    return f"{token} | {status} | {conf * 100:.2f}%"


def _find_original_from_gradcam(folder: Path, gradcam_path: Path) -> Optional[Path]:
    """
    gradcam file: <stem>_gradcam.png
    original:     <stem>.png  (prefer), fallback .jpg/.jpeg
    """
    name = gradcam_path.name
    if not name.endswith("_gradcam.png"):
        return None
    stem = name[: -len("_gradcam.png")]  # <stem>
    for ext in [".png", ".jpg", ".jpeg"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _concat_horiz(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    """
    Concatenate two images horizontally (same height by resizing right to left height).
    """
    left = img_left
    right = img_right
    if right.size[1] != left.size[1]:
        new_w = int(right.size[0] * (left.size[1] / right.size[1]))
        right = right.resize((new_w, left.size[1]), Image.Resampling.BILINEAR)

    out = Image.new("RGB", (left.size[0] + right.size[0], left.size[1]))
    out.paste(left, (0, 0))
    out.paste(right, (left.size[0], 0))
    return out


def viz_sgm_predict(
    root_folder: str,
    figsize=(18, 5),
    compare: bool = False,
    max_plot: int = 20,
    output_folder: Optional[str] = None,
) -> None:
    """
    - Parse patterns ONLY from *_gradcam.png filenames
    - 1-row layout: CC_R, CC_L, MLO_R, MLO_L
    - Title compact (no .png)
    - If compare=True: Positive tiles show [original | gradcam] merged horizontally
    - Save predictions to sgm_predict.csv with columns: image_id, label, confidence_score, ground_truth
    - NEW: if prediction matches ground_truth -> title GREEN, else RED
    """

    def _mpl_safe_text(s: str) -> str:
        # Matplotlib treats '$' as mathtext delimiter -> escape it
        return str(s).replace("$", r"\$")

    def _to_int(x) -> Optional[int]:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    def _to_float(x) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _read_metadata_csv(
        path: Path,
    ) -> Dict[str, Tuple[Optional[int], Optional[float], Optional[int]]]:
        """
        Return map: stem -> (pred_label, conf, ground_truth)
        Supports common column names:
          - image_id / id / stem
          - label / pred / y_pred
          - confidence_score / conf / confidence / score
          - ground_truth / gt / y_true / target
        """
        mp: Dict[str, Tuple[Optional[int], Optional[float], Optional[int]]] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id") or row.get("id") or row.get("stem")
                if not image_id:
                    continue
                stem = Path(str(image_id)).stem

                pred = row.get("label") or row.get("pred") or row.get("y_pred")
                conf = (
                    row.get("confidence_score")
                    or row.get("conf")
                    or row.get("confidence")
                    or row.get("score")
                )
                gt = (
                    row.get("ground_truth")
                    or row.get("gt")
                    or row.get("y_true")
                    or row.get("target")
                )

                mp[stem] = (_to_int(pred), _to_float(conf), _to_int(gt))
        return mp

    root = Path(root_folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    if output_folder:
        out_dir = Path(output_folder).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    folders = sorted(
        [p for p in root.rglob("*") if p.is_dir() and (p / "metadata.csv").exists()]
    )
    if not folders:
        raise FileNotFoundError(f"No subfolder with metadata.csv found under: {root}")

    # Đếm tổng số plot nếu chạy full và có output_folder để dùng tqdm
    total_plots = 0
    if max_plot is None and output_folder:
        for folder in folders:
            csv_map = None
            gradcams = sorted(folder.glob("*_gradcam.png"))
            groups: Dict[str, Dict[Tuple[str, str], Dict]] = {}
            for p in gradcams:
                parsed = _parse_gradcam_name(p.name)
                if parsed is None:
                    continue
                base, view, side, idx = parsed
                key = (view, side)
                stem = p.name[: -len("_gradcam.png")]
                if csv_map is None:
                    csv_map = _read_metadata_csv(folder / "metadata.csv")
                pred_conf_gt = csv_map.get(stem)
                label = pred_conf_gt[0] if pred_conf_gt else None
                conf = pred_conf_gt[1] if pred_conf_gt else None
                gt = pred_conf_gt[2] if pred_conf_gt else None
                cand = {"path": p, "label": label, "conf": conf, "gt": gt, "idx": idx}
                groups.setdefault(base, {})
                if key not in groups[base]:
                    groups[base][key] = cand
                else:
                    prev = groups[base][key]
                    prev_conf = prev["conf"]
                    cand_conf = cand["conf"]
                    if prev_conf is None and cand_conf is not None:
                        groups[base][key] = cand
                    elif (
                        prev_conf is not None
                        and cand_conf is not None
                        and cand_conf > prev_conf
                    ):
                        groups[base][key] = cand
            total_plots += len(groups)

    plot_count = 0
    folder_iter = folders
    if max_plot is None and output_folder:
        folder_iter = tqdm(folders, desc="Folders", leave=True)
    for folder in folder_iter:
        csv_map = _read_metadata_csv(folder / "metadata.csv")
        gradcams = sorted(folder.glob("*_gradcam.png"))
        if not gradcams:
            print(f"[WARN] No *_gradcam.png in: {folder}")
            continue

        groups: Dict[str, Dict[Tuple[str, str], Dict]] = {}

        for p in gradcams:
            parsed = _parse_gradcam_name(p.name)
            if parsed is None:
                continue
            base, view, side, idx = parsed
            key = (view, side)

            stem = p.name[: -len("_gradcam.png")]  # e.g., xxx_CC_R_2
            pred_conf_gt = csv_map.get(stem)
            label = pred_conf_gt[0] if pred_conf_gt else None
            conf = pred_conf_gt[1] if pred_conf_gt else None
            gt = pred_conf_gt[2] if pred_conf_gt else None

            cand = {"path": p, "label": label, "conf": conf, "gt": gt, "idx": idx}

            groups.setdefault(base, {})
            if key not in groups[base]:
                groups[base][key] = cand
            else:
                prev = groups[base][key]
                prev_conf = prev["conf"]
                cand_conf = cand["conf"]
                if prev_conf is None and cand_conf is not None:
                    groups[base][key] = cand
                elif (
                    prev_conf is not None
                    and cand_conf is not None
                    and cand_conf > prev_conf
                ):
                    groups[base][key] = cand

        if not groups:
            print(f"[WARN] No CC/MLO L/R gradcam pattern matched in: {folder}")
            continue

        pred_rows = []

        # Collect all figures for this folder if output_folder is set
        folder_figs = []
        folder_titles = []

        base_ids = sorted(groups.keys())
        # Nếu dùng tqdm cho từng base_id khi chạy full và có output_folder
        if max_plot is None and output_folder:
            base_id_iter = tqdm(base_ids, desc=f"{folder.name}", leave=False)
        else:
            base_id_iter = base_ids

        for base_id in base_id_iter:
            if max_plot is not None and plot_count >= max_plot:
                return
            g = groups[base_id]

            # derive blank size from first available image
            first_img = None
            for view, side in VIEW_ORDER:
                item = g.get((view, side))
                if item:
                    first_img = _load_rgb(item["path"])
                    break
            blank_size = first_img.size if first_img else (512, 512)

            imgs: List[Image.Image] = []
            titles: List[str] = []
            title_colors: List[str] = []

            for view, side in VIEW_ORDER:
                item = g.get((view, side))
                if item is None:
                    imgs.append(_blank_image(blank_size))
                    titles.append(f"{view}_{side} | Missing")
                    title_colors.append("black")
                    continue

                grad_img = _load_rgb(item["path"])
                label = item["label"]
                conf = item["conf"]
                gt = item.get("gt")

                pred_rows.append(
                    {
                        "image_id": item["path"].stem.replace("_gradcam", ""),
                        "label": label,
                        "confidence_score": conf,
                        "ground_truth": gt,
                    }
                )

                # compare view
                if compare:
                    orig_path = _find_original_from_gradcam(folder, item["path"])
                    show_concat = (label == 1) or (gt == 1)
                    if orig_path and orig_path.exists():
                        orig_img = _load_rgb(orig_path)
                        if show_concat:
                            imgs.append(
                                _concat_horiz(orig_img, grad_img)
                            )  # Positive: [orig | gradcam]
                        else:
                            imgs.append(orig_img)  # Negative: show original only
                    else:
                        imgs.append(grad_img)  # fallback if original missing
                else:
                    imgs.append(grad_img)

                # Title text
                titles.append(_compact_title(view, side, label, conf))

                # Title color: green if correct, red if wrong (only when both exist)
                if label is not None and gt is not None:
                    title_colors.append("green" if int(label) == int(gt) else "red")
                else:
                    title_colors.append("black")

            fig, axes = plt.subplots(1, 4, figsize=figsize)
            fig.suptitle(_mpl_safe_text(f"{folder.name} | {base_id}"), fontsize=14)
            for ax, im, ttl, col in zip(axes, imgs, titles, title_colors):
                ax.imshow(im)
                ax.set_title(_mpl_safe_text(ttl), fontsize=10, color=col)
                ax.axis("off")
            plt.tight_layout()

            if out_dir:
                # Save to output folder, one image per base_id
                out_name = f"{folder.name}_{base_id}.png"
                out_path = out_dir / out_name
                fig.savefig(str(out_path))
                plt.close(fig)
            else:
                plt.show()
            plot_count += 1
            if max_plot is None and output_folder and plot_count >= total_plots:
                break
        # Save predictions to CSV (uncomment if you want)
        # out_csv = folder / "sgm_predict.csv"
        # with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        #     writer = csv.DictWriter(
        #         fout,
        #         fieldnames=["image_id", "label", "confidence_score", "ground_truth"]
        #     )
        #     writer.writeheader()
        #     writer.writerows(pred_rows)
        # print(f"[INFO] Saved predictions to: {out_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize SGM predictions")
    parser.add_argument(
        "root_folder",
        type=str,
        help="Root folder chứa các subfolder với metadata.csv và *_gradcam.png",
    )
    parser.add_argument(
        "--compare", action="store_true", help="So sánh [original|gradcam] cho positive"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(18, 5),
        metavar=("W", "H"),
        help="Kích thước figure matplotlib (mặc định: 18 5)",
    )
    parser.add_argument(
        "--max_plot",
        type=str,
        default="20",
        help="Số lượng plot tối đa (mặc định: 20), hoặc 'all' để chạy hết",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Thư mục để lưu các ảnh đã plot (mỗi folder thành 1 ảnh)",
    )

    args = parser.parse_args()
    # Handle max_plot: allow 'all'
    if args.max_plot.lower() == "all":
        max_plot = None
    else:
        try:
            max_plot = int(args.max_plot)
        except Exception:
            max_plot = 20

    viz_sgm_predict(
        args.root_folder,
        figsize=tuple(args.figsize),
        compare=args.compare,
        max_plot=max_plot,
        output_folder=args.output_folder,
    )
