from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)

# 처리할 영상 확장자
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ③ 터미널에서 영상 경로 및 실행 옵션 입력받기
def parse_args():
    parser = argparse.ArgumentParser()

    # 분석할 영상 파일 또는 영상 폴더 (기본값: data/day02/videos 폴더 전체)
    parser.add_argument(
        "--source",
        default=str(ROOT / "data" / "day02" / "videos"),
        help="영상 파일 하나 또는 영상이 들어 있는 폴더",
    )

    # LOW 관절을 판단할 Confidence 기준
    parser.add_argument("--kp-conf", type=float, default=0.50)

    # 최대 처리 Frame 수 (0이면 영상 전체)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="영상별 처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


# ④ 처리할 영상 목록 만들기
def collect_videos(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"경로가 없습니다: {source}")

    # 파일 하나를 지정한 경우
    if source.is_file():
        return [source]

    # 폴더를 지정한 경우: 안에 있는 영상 전부 (이름순, 결과 영상 *_pose 제외)
    videos = sorted(
        p for p in source.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.endswith("_pose")
    )

    if not videos:
        raise FileNotFoundError(f"폴더에 영상이 없습니다: {source}")

    return videos


# ⑤ 영상 하나를 처리하는 함수
def process_video(source: Path, model, device, config, args) -> None:

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    # ⑥ 영상 FPS·해상도 및 첫 Frame 확인
    fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps > 0 else 30.0

    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError("영상의 첫 Frame을 읽지 못했습니다.")

    height, width = frame.shape[:2]

    # ⑦ Pose 결과 MP4 저장 준비
    output_video = (
        ROOT / "reports" / "day02" / "videos" / f"{source.stem}_pose.mp4"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("결과 MP4를 생성할 수 없습니다.")

    # ⑧ Frame 품질 CSV와 Keypoint CSV 저장 준비
    tables_dir = ROOT / "reports" / "day02" / "tables"
    frame_table = tables_dir / f"{source.stem}_frame_quality.csv"
    keypoint_table = tables_dir / f"{source.stem}_keypoints.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    keypoint_rows = []
    frame_id = 0

    # ⑨ Frame별 YOLO Pose 추론 및 결과 영상 저장
    while success:
        frame_id += 1

        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )
        result = results[0]

        writer.write(result.plot())

        # ⑩ 검출된 사람 수 및 품질 기본값 설정
        people = len(result.boxes) if result.boxes is not None else 0

        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        # ⑪ 대표 사람의 관절 Confidence 분석
        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.xy is not None
            and result.keypoints.conf is not None
            and len(result.boxes) > 0
        ):
            box_conf = result.boxes.conf.cpu().numpy()
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy()

            main_index = int(np.argmax(box_conf))
            main_xy = xy[main_index]
            main_conf = conf[main_index]

            mean_kp_conf = float(np.mean(main_conf))
            low_keypoints = int(np.sum(main_conf < args.kp_conf))

            lowest_id = int(np.argmin(main_conf))
            lowest_name = COCO_KEYPOINT_NAMES[lowest_id]
            lowest_conf = float(main_conf[lowest_id])

            # ⑫ 현재 Frame의 17개 Keypoint 상세 정보 기록
            for keypoint_id, name in enumerate(COCO_KEYPOINT_NAMES):
                x, y = main_xy[keypoint_id]
                confidence = main_conf[keypoint_id]

                keypoint_rows.append(
                    {
                        "frame_id": frame_id,
                        "time_sec": round((frame_id - 1) / output_fps, 3),
                        "keypoint_id": keypoint_id,
                        "keypoint_name": name,
                        "pixel_x": round(float(x), 2),
                        "pixel_y": round(float(y), 2),
                        "confidence": round(float(confidence), 4),
                    }
                )

        # ⑬ 현재 Frame의 Pose 품질 요약 기록
        frame_rows.append(
            {
                "frame_id": frame_id,
                "time_sec": round((frame_id - 1) / output_fps, 3),
                "people": people,
                "mean_keypoint_conf": round(mean_kp_conf, 4),
                "low_keypoint_count": low_keypoints,
                "lowest_keypoint": lowest_name,
                "lowest_keypoint_conf": round(lowest_conf, 4),
                "keypoint_threshold": args.kp_conf,
            }
        )

        # ⑭ 진행 상황 확인 및 다음 Frame 읽기
        if frame_id % 30 == 0:
            print(f"  [{source.name}] Processed frame: {frame_id}")

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

        success, frame = capture.read()

    # ⑮ 영상 자원 해제 및 Frame 품질 CSV 저장
    capture.release()
    writer.release()

    with frame_table.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "people",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    # ⑯ Keypoint 상세 CSV 저장 및 결과 출력
    with keypoint_table.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "keypoint_id",
                "keypoint_name",
                "pixel_x",
                "pixel_y",
                "confidence",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(keypoint_rows)

    print(f"  Frames processed: {frame_id}")
    print(f"  Pose video: {output_video}")
    print(f"  Frame table: {frame_table}")
    print(f"  Keypoint table: {keypoint_table}")


def main() -> None:
    args = parse_args()

    # 처리할 영상 목록
    videos = collect_videos(Path(args.source))
    print(f"처리할 영상 {len(videos)}개")

    # 설정 및 모델은 한 번만 불러오기
    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    failed = []

    # 영상별로 순서대로 처리
    for index, video in enumerate(videos, start=1):
        print()
        print(f"[{index}/{len(videos)}] {video.name}")

        try:
            process_video(video, model, device, config, args)
        except Exception as error:
            # 한 영상이 실패해도 나머지는 계속 처리
            print(f"  실패: {error}")
            failed.append(video.name)

    # 전체 결과 요약
    print()
    print(f"완료: {len(videos) - len(failed)}/{len(videos)}")
    if failed:
        print("실패한 영상:", ", ".join(failed))


if __name__ == "__main__":
    main()