import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import cv2
import numpy as np


def safe_convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    elif image.mode in ["L", "P"]:
        return image.convert("RGB")
    elif image.mode in ["I;16", "I", "F"]:
        # Normalize pixel values to [0,255] and convert to uint8
        img_8bit = image.point(
            lambda x: x * (255.0 / 65535.0) if image.mode == "I;16" else x
        )
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


class DMImagePreprocessor(object):
    """Class for preprocessing images in the DM challenge"""

    def __init__(self):
        pass

    def select_largest_obj(
        self,
        img_bin,
        lab_val=255,
        fill_holes=False,
        smooth_boundary=False,
        kernel_size=15,
    ):
        n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
            img_bin, connectivity=8, ltype=cv2.CV_32S
        )
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val
        if fill_holes:
            bkg_locs = np.where(img_labeled == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)
            largest_mask = largest_mask + holes_mask
        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)
        return largest_mask

    @staticmethod
    def max_pix_val(dtype):
        if dtype == np.dtype("uint8"):
            maxval = 2**8 - 1
        elif dtype == np.dtype("uint16"):
            maxval = 2**16 - 1
        else:
            raise Exception("Unknown dtype found in input image array")
        return maxval

    def suppress_artifacts(
        self,
        img,
        global_threshold=0.05,
        fill_holes=False,
        smooth_boundary=True,
        kernel_size=15,
    ):
        maxval = self.max_pix_val(img.dtype)
        if global_threshold < 1.0:
            low_th = int(img.max() * global_threshold)
        else:
            low_th = int(global_threshold)
        _, img_bin = cv2.threshold(img, low_th, maxval=maxval, type=cv2.THRESH_BINARY)
        breast_mask = self.select_largest_obj(
            img_bin,
            lab_val=maxval,
            fill_holes=True,
            smooth_boundary=True,
            kernel_size=kernel_size,
        )
        img_suppr = cv2.bitwise_and(img, breast_mask)
        return (img_suppr, breast_mask)

    @classmethod
    def segment_breast(cls, img, low_int_threshold=0.05, crop=True):
        img_8u = (img.astype("float32") / img.max() * 255).astype("uint8")
        if low_int_threshold < 1.0:
            low_th = int(img_8u.max() * low_int_threshold)
        else:
            low_th = int(low_int_threshold)
        _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
        ver = (cv2.__version__).split(".")
        if int(ver[0]) < 3:
            contours, _ = cv2.findContours(
                img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            contours, _ = cv2.findContours(
                img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        cont_areas = [cv2.contourArea(cont) for cont in contours]
        idx = np.argmax(cont_areas)
        breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1)
        img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
        x, y, w, h = cv2.boundingRect(contours[idx])
        if crop:
            img_breast_only = img_breast_only[y : y + h, x : x + w]
        return img_breast_only, (x, y, w, h)


def SGM_preprocess(
    input_root: str,
    model_path: str,
    output_root: Optional[str] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    class_filter: Optional[List[int]] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> Tuple[Path, Dict[str, Tuple[int, int, int, int]]]:
    """
    Folder A: A/subfolder/picture.(png/jpg/jpeg)
    -> Output: A_preprocess/subfolder/picture.png  (always PNG)
    Behavior: crop ONLY the best (highest confidence) YOLO bbox.

    If no detection passes thresholds, the image is skipped.

    Returns:
        (output_root_path, crop_info_dict)
        crop_info_dict: {relative_path: (crop_x1, crop_y1, crop_x2, crop_y2)}
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
    fallback_basic = 0

    # Dictionary to store crop information: {relative_path: (x1, y1, x2, y2)}
    crop_info_dict = {}

    preprocessor = DMImagePreprocessor()
    pbar = tqdm(image_paths, desc="SGM_preprocess", unit="img", disable=(n == 0))
    for idx, img_path in enumerate(pbar, start=1):
        rel = img_path.relative_to(input_root_p)
        rel_str = str(rel).replace("\\", "/")

        try:
            res = model.predict(
                source=str(img_path),
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
            )[0]

            boxes = res.boxes
            yolo_detected = False
            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                clss = (
                    boxes.cls.cpu().numpy().astype(int)
                    if boxes.cls is not None
                    else None
                )

                keep = list(range(len(xyxy)))
                if class_filter is not None and clss is not None:
                    wanted = set(class_filter)
                    keep = [i for i in keep if int(clss[i]) in wanted]
                if keep:
                    best_i = (
                        keep[0]
                        if confs is None
                        else max(keep, key=lambda i: float(confs[i]))
                    )
                    img = Image.open(img_path)
                    img = safe_convert_to_rgb(img)
                    w, h = img.size

                    x1, y1, x2, y2 = xyxy[best_i]
                    clipped = _clip_box_xyxy(x1, y1, x2, y2, w, h)
                    if clipped is not None:
                        crop = img.crop(clipped)
                        out_dir = output_root_p / rel.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{img_path.stem}.png"
                        if out_path.exists() and not overwrite:
                            skipped_exists += 1
                            # Still record crop info even if skipped
                            crop_info_dict[rel_str] = clipped
                            pbar.set_postfix_str(
                                f"ok={processed} no_det={skipped_no_det} basic={fallback_basic} exists={skipped_exists} fail={failed}"
                            )
                            continue
                        crop.save(out_path, format="PNG")
                        processed += 1
                        yolo_detected = True
                        # Store crop bbox info
                        crop_info_dict[rel_str] = clipped

            # Nếu YOLO không detect được hoặc bbox không hợp lệ, dùng basic crop
            if not yolo_detected:
                if verbose:
                    print(f"\n[INFO] YOLO không detect được: {rel}, dùng basic crop...")
                try:
                    img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img_cv is None:
                        raise Exception("Không đọc được ảnh bằng cv2")
                    img_cropped, (x_c, y_c, w_c, h_c) = preprocessor.segment_breast(
                        img_cv, crop=True
                    )
                    out_dir = output_root_p / rel.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{img_path.stem}.png"
                    if out_path.exists() and not overwrite:
                        skipped_exists += 1
                        # Store crop info from basic crop
                        crop_info_dict[rel_str] = (x_c, y_c, x_c + w_c, y_c + h_c)
                        pbar.set_postfix_str(
                            f"ok={processed} no_det={skipped_no_det} basic={fallback_basic} exists={skipped_exists} fail={failed}"
                        )
                        continue
                    cv2.imwrite(str(out_path), img_cropped)
                    fallback_basic += 1
                    # Store crop bbox info from basic crop
                    crop_info_dict[rel_str] = (x_c, y_c, x_c + w_c, y_c + h_c)
                    if verbose:
                        print(f"[INFO] Đã dùng basic crop: {rel}")
                except Exception as e_basic:
                    # Nếu basic crop cũng fail, lưu ảnh gốc (no crop = full image)
                    if verbose:
                        print(
                            f"\n[WARN] Basic crop fail: {rel} - {e_basic}, lưu ảnh gốc..."
                        )
                    try:
                        img = Image.open(img_path)
                        img = safe_convert_to_rgb(img)
                        w, h = img.size
                        out_dir = output_root_p / rel.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{img_path.stem}.png"
                        if out_path.exists() and not overwrite:
                            skipped_exists += 1
                            # No crop = full image
                            crop_info_dict[rel_str] = (0, 0, w, h)
                            pbar.set_postfix_str(
                                f"ok={processed} no_det={skipped_no_det} basic={fallback_basic} exists={skipped_exists} fail={failed}"
                            )
                            continue
                        img.save(out_path, format="PNG")
                        skipped_no_det += 1
                        # No crop = full image
                        crop_info_dict[rel_str] = (0, 0, w, h)
                    except Exception as e_save:
                        failed += 1
                        if verbose:
                            print(f"[ERROR] Không thể lưu ảnh gốc: {rel} - {e_save}")

            pbar.set_postfix_str(
                f"ok={processed} no_det={skipped_no_det} basic={fallback_basic} exists={skipped_exists} fail={failed}"
            )

            if verbose and (idx % 50 == 0):
                print(
                    f"[PROGRESS] {idx}/{n} processed={processed}, no_det={skipped_no_det}, basic={fallback_basic}, failed={failed}"
                )

        except Exception as e:
            failed += 1
            pbar.set_postfix_str(
                f"ok={processed} no_det={skipped_no_det} basic={fallback_basic} exists={skipped_exists} fail={failed}"
            )
            if verbose:
                print(f"[ERROR] Failed on {rel}: {e}")

    if verbose:
        print("\n[SUMMARY]")
        print(f"  processed images (YOLO): {processed}")
        print(f"  fallback basic crop    : {fallback_basic}")
        print(f"  skipped (no det/orig)  : {skipped_no_det}")
        print(f"  skipped (exists)       : {skipped_exists}")
        print(f"  failed                 : {failed}")

    return output_root_p, crop_info_dict
