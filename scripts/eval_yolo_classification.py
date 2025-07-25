import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO classification model")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset directory in YOLO format",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate (val or test)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data_dir, imgsz=args.img_size, split=args.split)
    print(metrics)


if __name__ == "__main__":
    main()
