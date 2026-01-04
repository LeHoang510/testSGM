import os
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def safe_convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    elif image.mode in ["L", "P"]:
        return image.convert("RGB")
    elif image.mode in ["I;16", "I", "F"]:
        # Normalize pixel values to [0,255] and convert to uint8
        img_8bit = image.point(lambda x: x * (255.0 / 65535.0) if image.mode == "I;16" else x)
        return img_8bit.convert("L").convert("RGB")
    else:
        return image.convert("RGB")


def _list_images(root: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    files: List[Path] = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(Path(dp) / f)
    return files


def _clip_box_xyxy(x1, y1, x2, y2, w, h) -> Optional[Tuple[int, int, int, int]]:
    # Clip to valid image region
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def SGM_preprocess(
    input_root: str,
    model_path: str,
    output_root: Optional[str] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    class_filter: Optional[List[int]] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Folder A: A/subfolder/picture.(png/jpg/jpeg)
    -> Output: A_preprocess/subfolder/picture.png  (always PNG)
    Behavior: crop ONLY the best (highest confidence) YOLO bbox.

    If no detection passes thresholds, the image is skipped.
    """
    input_root_p = Path(input_root).expanduser().resolve()
    output_root_p = (
        input_root_p.parent / f"{input_root_p.name}_preprocess"
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    output_root_p.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(model_path)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"File not found: {model_path}")

    print("File exists:", model_path)


    model = YOLO(model_path, verbose=False)

    image_paths = _list_images(input_root_p)
    n = len(image_paths)

    if verbose:
        print(f"[INFO] Found {n} images under: {input_root_p}")
        print(f"[INFO] Output folder: {output_root_p}")
        print(f"[INFO] conf_thres={conf_thres}, iou_thres={iou_thres}")

    processed = 0
    skipped_no_det = 0
    skipped_exists = 0
    failed = 0

    pbar = tqdm(image_paths, desc="SGM_preprocess", unit="img", disable=(n == 0))
    for idx, img_path in enumerate(pbar, start=1):
        rel = img_path.relative_to(input_root_p)

        try:
            res = model.predict(
                source=str(img_path),
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
            )[0]

            boxes = res.boxes
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                skipped_no_det += 1
                pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None

            keep = list(range(len(xyxy)))
            if class_filter is not None and clss is not None:
                wanted = set(class_filter)
                keep = [i for i in keep if int(clss[i]) in wanted]
            if not keep:
                skipped_no_det += 1
                pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")
                continue

            best_i = keep[0] if confs is None else max(keep, key=lambda i: float(confs[i]))

            img = Image.open(img_path)
            img = safe_convert_to_rgb(img)
            w, h = img.size

            x1, y1, x2, y2 = xyxy[best_i]
            clipped = _clip_box_xyxy(x1, y1, x2, y2, w, h)
            if clipped is None:
                skipped_no_det += 1
                pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")
                continue

            crop = img.crop(clipped)

            out_dir = output_root_p / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{img_path.stem}.png"
            if out_path.exists() and not overwrite:
                skipped_exists += 1
                pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")
                continue

            crop.save(out_path, format="PNG")
            processed += 1

            pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")

            if verbose and (idx % 50 == 0):
                print(f"[PROGRESS] {idx}/{n} processed={processed}, no_det={skipped_no_det}, failed={failed}")

        except Exception as e:
            failed += 1
            pbar.set_postfix_str(f"ok={processed} no_det={skipped_no_det} exists={skipped_exists} fail={failed}")
            if verbose:
                print(f"[ERROR] Failed on {rel}: {e}")

    if verbose:
        print("\n[SUMMARY]")
        print(f"  processed images : {processed}")
        print(f"  skipped (no det) : {skipped_no_det}")
        print(f"  skipped (exists) : {skipped_exists}")
        print(f"  failed           : {failed}")

    return output_root_p