from huggingface_hub import hf_hub_download
from src.pre_process import SGM_preprocess
from src.processing import SGMpredict_folder
import argparse
import os


def main(dataset_folder, compute_folder, output_folder, broi_model_path, pretrained_model_path, is_cropped_data, ground_truth_path):
    # Pre Processing. If data is not cropped => Crop
    pre_process_folder = os.path.join(compute_folder, "cropped")
    if not is_cropped_data:
        SGM_preprocess(
            input_root=dataset_folder,
            output_root=pre_process_folder,
            model_path=broi_model_path,
            # conf_thres defaults to 0.25
        )

    # Prediction
    SGMpredict_folder(
        input_root=pre_process_folder,
        model_path=pretrained_model_path,
        output_root=output_folder,
        ground_truth_path=ground_truth_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--compute_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--broi_model_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--is_cropped_data", action="store_true")
    parser.add_argument("--ground_truth_path", type=str, required=False)
    args = parser.parse_args()

    main(args.dataset_folder, args.compute_folder, args.output_folder, args.broi_model_path, args.pretrained_model_path, args.is_cropped_data, args.ground_truth_path)