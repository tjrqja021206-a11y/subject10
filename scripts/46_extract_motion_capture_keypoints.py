from __future__ import annotations

# ====================================================
# 46번. Motion Capture 영상 → 스켈레톤(Keypoint) CSV
# ====================================================
# 입력: data/motion_capture/S01/S01_A001_T01.mp4 ...
# 출력: data/motion_capture_keypoints/S01/S01_A001_T01.csv
#       data/motion_capture_keypoints/manifest_S01.csv
#
# - Frame별 처리와 CSV 컬럼(95개)은 26번 코드의 process_clip()을
#   그대로 재사용하므로 day05 Keypoint CSV와 형식이 같습니다.
# - Motion Capture 동작(A001~A017)은 day05의 action_classes.csv와
#   다르므로 configs/motion_capture_action_classes.csv를 사용합니다.
# - T01/T02는 같은 동작을 다른 방향에서 찍은 영상입니다.
#   Manifest에 orientation(portrait/landscape)을 함께 남깁니다.

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    load_config,
    resolve_device,
)

# 파일명이 숫자로 시작하므로 importlib로 26번 모듈을 불러옵니다.
_spec = importlib.util.spec_from_file_location(
    "extract_keypoint_dataset",
    ROOT / "scripts" / "26_extract_keypoint_dataset.py",
)
extract26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract26)


ACTION_FILE = ROOT / "configs" / "motion_capture_action_classes.csv"


def load_action_map() -> dict[str, str]:
    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        return {
            row["action_id"].strip().upper(): row["action_label"].strip()
            for row in csv.DictReader(file)
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        default="S01",
    )
    parser.add_argument(
        "--video-root",
        default=str(ROOT / "data" / "motion_capture"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "motion_capture_keypoints"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject = args.subject.upper()
    video_dir = Path(args.video_root) / subject
    output_root = Path(args.output_root)

    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    clips = sorted(video_dir.glob("*.mp4"))
    if not clips:
        raise RuntimeError(f"MP4가 없습니다: {video_dir}")

    # 파일명 규칙 검사 및 Action ID 수집
    action_ids = set()
    for clip in clips:
        match = extract26.FILENAME_PATTERN.match(clip.name)
        if match is None:
            raise ValueError(f"파일명 규칙 오류: {clip.name}")
        action_ids.add(match.group(2).upper())

    # 동작 이름은 motion_capture_action_classes.csv에서 읽습니다.
    action_map = load_action_map()
    missing = sorted(action_ids - set(action_map))
    if missing:
        raise RuntimeError(
            f"{ACTION_FILE.name}에 없는 Action ID: {missing}"
        )

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(str(ROOT / config["pose_model"]))

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Subject: {subject}")
    print(f"Clips: {len(clips)} / Actions: {len(action_map)}")
    print(f"Device: {device}")
    print()

    manifest_rows = []
    for index, clip in enumerate(clips, start=1):
        print(f"[{index}/{len(clips)}] {clip.name}")
        item = extract26.process_clip(
            model=model,
            clip_path=clip,
            action_map=action_map,
            config=config,
            device=device,
            output_root=output_root,
        )
        # 26번 코드는 S01/<label>/ 하위에 저장하므로
        # 영상 폴더와 같은 S01/ 바로 아래로 옮깁니다.
        nested_csv = ROOT / item["keypoint_csv"]
        flat_csv = output_root / subject / nested_csv.name
        nested_csv.replace(flat_csv)
        if not any(nested_csv.parent.iterdir()):
            nested_csv.parent.rmdir()
        item["keypoint_csv"] = str(flat_csv.relative_to(ROOT))
        item["orientation"] = (
            "portrait"
            if item["height"] >= item["width"]
            else "landscape"
        )
        manifest_rows.append(item)
        print(
            f"  pose_ratio={item['pose_ratio']} "
            f"mean_conf={item['mean_keypoint_conf']} "
            f"{item['orientation']}"
        )

    manifest_path = output_root / f"manifest_{subject}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Motion capture keypoint extraction: COMPLETE")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
