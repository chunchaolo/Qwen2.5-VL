import os
import json
import random
import argparse
from pathlib import Path

import cv2

LABELS = [
    "Pass",
    "2-pt Shot",
    "3-pt Shot",
    "Free-throw",
    "Off. Rebound",
    "Def. Rebound",
    "Steal",
    "Block",
    "Assist",
    "Turnover",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert video conversation dataset to YOLO classification format",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file with video/conversation annotations",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save converted dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of validation data split",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Interval of frames to extract from video",
    )
    return parser.parse_args()


def parse_label(gpt_value: str) -> str:
    """Parse gpt response JSON and return the label string."""
    data = json.loads(gpt_value.strip())
    label = data.get("label")
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    return label


def extract_frames(video_path: str, out_dir: Path, interval: int):
    """Extract frames from video and save to out_dir."""
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            img_name = f"{Path(video_path).stem}_{idx}.jpg"
            cv2.imwrite(str(out_dir / img_name), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def process_split(data, split_dir: Path, interval: int):
    for item in data:
        video = item["video"]
        gpt_reply = next((c["value"] for c in item["conversations"] if c["from"] == "gpt"), None)
        if gpt_reply is None:
            continue
        try:
            label = parse_label(gpt_reply)
        except Exception as e:
            print(f"Skip {video}: {e}")
            continue
        out_dir = split_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(video, out_dir, interval)


def main():
    args = parse_args()
    out_root = Path(args.output)
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    val_size = int(len(data) * args.val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]

    process_split(train_data, train_dir, args.frame_interval)
    process_split(val_data, val_dir, args.frame_interval)


if __name__ == "__main__":
    main()
