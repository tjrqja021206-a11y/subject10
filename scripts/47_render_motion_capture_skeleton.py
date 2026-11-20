from __future__ import annotations

# ====================================================
# 47번. 스켈레톤 CSV를 원본 영상 위에 그려서 확인
# ====================================================
# 입력: data/motion_capture/S01/S01_A001_T01.mp4
#       data/motion_capture_keypoints/S01/S01_A001_T01.csv (46번 결과)
# 출력: data/motion_capture_overlay/S01/S01_A001_T01.mp4
#
# - 다시 추론하지 않고 CSV에 저장된 좌표를 그대로 그립니다.
#   (= CSV가 올바른지 눈으로 확인하는 용도)
# - 색: 왼쪽 관절/뼈 = 파랑, 오른쪽 = 주황, 몸통·얼굴 = 초록
#   Confidence < 0.5 관절은 빨간 X로 표시합니다.
# - 화면 위에 Action·Take·Frame·검출 인원 수를 표시하며
#   사람이 2명 이상 잡힌 Frame은 테두리를 빨갛게 칠합니다.
# - --sheet-dir를 주면 Clip마다 8칸 요약 이미지(JPG)도 저장합니다.

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import (  # noqa: E402
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
)

ACTION_FILE = ROOT / "configs" / "motion_capture_action_classes.csv"

LEFT = (255, 128, 0)     # BGR 파랑
RIGHT = (0, 140, 255)    # BGR 주황
CENTER = (80, 200, 80)   # BGR 초록
LOW = (0, 0, 255)        # BGR 빨강
CONF_THRESHOLD = 0.5


def side_color(name: str):
    if name.startswith("left_"):
        return LEFT
    if name.startswith("right_"):
        return RIGHT
    return CENTER


def edge_color(a: int, b: int):
    na, nb = COCO_KEYPOINT_NAMES[a], COCO_KEYPOINT_NAMES[b]
    if na.startswith("left_") and nb.startswith("left_"):
        return LEFT
    if na.startswith("right_") and nb.startswith("right_"):
        return RIGHT
    return CENTER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="S01")
    parser.add_argument(
        "--video-root",
        default=str(ROOT / "data" / "motion_capture"),
    )
    parser.add_argument(
        "--keypoint-root",
        default=str(ROOT / "data" / "motion_capture_keypoints"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "motion_capture_overlay"),
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=960,
        help="출력 영상의 긴 변 길이(px)",
    )
    parser.add_argument("--sheet-dir", default=None)
    parser.add_argument(
        "--clip",
        default=None,
        help="한 Clip만 처리 (예: S01_A001_T01)",
    )
    return parser.parse_args()


def load_descriptions() -> dict[str, str]:
    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        return {
            row["action_id"]: row["action_label"]
            for row in csv.DictReader(file)
        }


def draw_pose(frame, row, scale: float, thickness: int) -> None:
    if int(row["pose_detected"]) != 1:
        return

    points = []
    for name in COCO_KEYPOINT_NAMES:
        x = row[f"{name}_x_px"] * scale
        y = row[f"{name}_y_px"] * scale
        conf = row[f"{name}_conf"]
        points.append((int(round(x)), int(round(y)), float(conf)))

    for a, b in COCO_SKELETON:
        xa, ya, ca = points[a]
        xb, yb, cb = points[b]
        if min(ca, cb) < CONF_THRESHOLD:
            continue
        cv2.line(
            frame, (xa, ya), (xb, yb),
            edge_color(a, b), thickness, cv2.LINE_AA,
        )

    radius = thickness + 1
    for name, (x, y, conf) in zip(COCO_KEYPOINT_NAMES, points):
        if conf < CONF_THRESHOLD:
            cv2.drawMarker(
                frame, (x, y), LOW, cv2.MARKER_TILTED_CROSS,
                radius * 3, 2,
            )
        else:
            cv2.circle(frame, (x, y), radius, side_color(name), -1)


def draw_text(frame, lines: list[str], scale_px: float) -> None:
    font_scale = max(0.45, scale_px / 900)
    line_h = int(26 * font_scale / 0.6)
    y = int(28 * font_scale / 0.6)
    # 글자 뒤에 어두운 띠를 깔아 배경과 상관없이 읽히게 합니다.
    band = frame[: y + line_h * (len(lines) - 1) + 10]
    band[:] = (band * 0.35).astype(band.dtype)
    for text in lines:
        cv2.putText(
            frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_h


def render_clip(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    label: str,
    max_side: int,
    sheet_path: Path | None,
) -> dict:
    data = pd.read_csv(csv_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_side / max(width, height))
    out_w, out_h = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    thickness = max(2, int(round(max(out_w, out_h) / 320)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    sheet_ids = set(
        np.linspace(0, len(data) - 1, 8).round().astype(int).tolist()
    )
    sheet_frames = []
    multi_person = 0

    for frame_id, row in data.iterrows():
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.resize(frame, (out_w, out_h))
        draw_pose(frame, row, scale, thickness)

        count = int(row["person_count"])
        if count > 1:
            multi_person += 1
            cv2.rectangle(
                frame, (0, 0), (out_w - 1, out_h - 1), LOW, 8,
            )

        draw_text(
            frame,
            [
                f"{row['clip_id']}  {label}",
                f"frame {frame_id}/{len(data) - 1}  "
                f"t={row['time_sec']:.2f}s  persons={count}",
            ],
            max(out_w, out_h),
        )
        writer.write(frame)

        if frame_id in sheet_ids:
            sheet_frames.append(frame.copy())

    capture.release()
    writer.release()

    if sheet_path is not None and sheet_frames:
        tile_h = 480
        tiles = [
            cv2.resize(f, (int(f.shape[1] * tile_h / f.shape[0]), tile_h))
            for f in sheet_frames
        ]
        rows = [
            np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)
        ]
        max_w = max(r.shape[1] for r in rows)
        rows = [
            cv2.copyMakeBorder(
                r, 0, 0, 0, max_w - r.shape[1],
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            for r in rows
        ]
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), np.vstack(rows))

    return {
        "frames": len(data),
        "multi_person_frames": multi_person,
    }


def main() -> None:
    args = parse_args()
    subject = args.subject.upper()
    video_dir = Path(args.video_root) / subject
    keypoint_dir = Path(args.keypoint_root) / subject
    output_dir = Path(args.output_root) / subject
    sheet_dir = Path(args.sheet_dir) if args.sheet_dir else None
    labels = load_descriptions()

    csv_paths = sorted(keypoint_dir.glob("*.csv"))
    if args.clip:
        csv_paths = [p for p in csv_paths if p.stem == args.clip]
    if not csv_paths:
        raise RuntimeError(f"CSV가 없습니다: {keypoint_dir}")

    for index, csv_path in enumerate(csv_paths, start=1):
        video_path = video_dir / f"{csv_path.stem}.mp4"
        action_id = csv_path.stem.split("_")[1]
        label = f"{action_id} {labels.get(action_id, '?')}"
        info = render_clip(
            video_path=video_path,
            csv_path=csv_path,
            output_path=output_dir / f"{csv_path.stem}.mp4",
            label=label,
            max_side=args.max_side,
            sheet_path=(
                sheet_dir / f"{csv_path.stem}.jpg" if sheet_dir else None
            ),
        )
        print(
            f"[{index}/{len(csv_paths)}] {csv_path.stem} {label} "
            f"frames={info['frames']} "
            f"multi_person={info['multi_person_frames']}"
        )

    print()
    print(f"Overlay videos: {output_dir}")


if __name__ == "__main__":
    main()
