from __future__ import annotations

# ====================================================
# 49번. 스켈레톤 CSV 좌우 반전 증강 (T02 → T02M)
# ====================================================
# T01=앞, T02=오른쪽, T03=뒤에서 찍은 영상입니다.
# 오른쪽(T02) 영상을 좌우로 뒤집으면 왼쪽에서 찍은 것과 비슷해지므로
# 찍지 않은 "왼쪽 방향" 데이터를 만들어 학습 데이터를 늘립니다.
#
# 영상을 다시 추론하지 않고 46번 CSV의 좌표만 바꿉니다.
#   x_px   → 영상 가로 - x_px
#   x_norm → 1 - x_norm
#   left_* ↔ right_* 관절 값 교환 (뒤집으면 왼손이 오른손 자리로 가므로)
#
# 입력: data/motion_capture_keypoints/<S>/<S>_A001_T02.csv
# 출력: data/motion_capture_keypoints_aug/<S>/<S>_A001_T02M.csv
#       (take 컬럼 = T02M, 증강 데이터는 학습에만 사용합니다)

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402

FIELDS = ("x_px", "y_px", "x_norm", "y_norm", "conf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        default=None,
        help="쉼표로 구분 (기본: 모든 Subject)",
    )
    parser.add_argument("--source-take", default="T02")
    parser.add_argument(
        "--keypoint-root",
        default=str(ROOT / "data" / "motion_capture_keypoints"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "motion_capture_keypoints_aug"),
    )
    return parser.parse_args()


def mirror_frame_table(df: pd.DataFrame, width: int) -> pd.DataFrame:
    out = df.copy()

    for name in COCO_KEYPOINT_NAMES:
        out[f"{name}_x_px"] = (width - df[f"{name}_x_px"]).round(3)
        out[f"{name}_x_norm"] = (1.0 - df[f"{name}_x_norm"]).round(6)

    mirrored = out.copy()
    for name in COCO_KEYPOINT_NAMES:
        if not name.startswith("left_"):
            continue
        other = "right_" + name[len("left_"):]
        for field in FIELDS:
            mirrored[f"{name}_{field}"] = out[f"{other}_{field}"]
            mirrored[f"{other}_{field}"] = out[f"{name}_{field}"]
    return mirrored


def main() -> None:
    args = parse_args()
    keypoint_root = Path(args.keypoint_root)
    output_root = Path(args.output_root)
    new_take = f"{args.source_take}M"

    if args.subjects:
        subjects = [s.strip().upper() for s in args.subjects.split(",")]
    else:
        subjects = sorted(
            p.name for p in keypoint_root.iterdir()
            if p.is_dir() and p.name.startswith("S")
        )

    total = 0
    for subject in subjects:
        manifest = pd.read_csv(keypoint_root / f"manifest_{subject}.csv")
        widths = dict(zip(manifest["clip_id"], manifest["width"]))
        out_dir = output_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(
            (keypoint_root / subject).glob(f"*_{args.source_take}.csv")
        ):
            df = pd.read_csv(csv_path)
            mirrored = mirror_frame_table(df, int(widths[csv_path.stem]))
            new_id = f"{csv_path.stem}M"
            mirrored["clip_id"] = new_id
            mirrored["take"] = new_take
            mirrored.to_csv(
                out_dir / f"{new_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            total += 1
        print(f"{subject}: {len(list(out_dir.glob('*.csv')))} mirrored clips")

    print(f"Mirrored: {total} clips -> {output_root}")


if __name__ == "__main__":
    main()
