import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO classification model")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset directory in YOLO format",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Initial model weights (classification)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size for training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data_dir, epochs=args.epochs, imgsz=args.img_size)


if __name__ == "__main__":
    main()
