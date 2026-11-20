from __future__ import annotations

# ====================================================
# 48번. Motion Capture 17개 동작 구분 테스트
# ====================================================
# T01과 T02는 같은 동작을 다른 방향에서 찍은 영상입니다.
# 다음 두 종류의 테스트를 합니다.
#   1) 방향 교차: T01로 학습 → T02로 테스트 (및 반대)
#   2) 사람 교차: 한 사람(양방향)으로 학습 → 다른 사람으로 테스트
#      (사람이 3명 이상이면 한 명씩 빼는 Leave-One-Subject-Out)
#
# - 입력: data/motion_capture_keypoints/<Subject>/*.csv (46번 결과)
# - Sliding Window: 2초 창을 0.25초 간격으로 자릅니다.
# - Feature 두 가지를 비교합니다.
#   seq    : day06과 같은 Frame Feature 38개(어깨너비 정규화 좌표 +
#            팔꿈치/무릎 각도)를 30 Step으로 Resample해 펼친 값
#   robust : 방향에 덜 민감한 Frame Feature 16개(관절 각도, 몸통 기울기,
#            엉덩이 기준 손·발·코 높이)의 창 안 통계(평균·표준편차·
#            최소·최대·평균 변화량). 좌우 쌍은 작은 값/큰 값으로 정렬하여
#            좌우 반전에 영향받지 않게 합니다.
# - 모델: Random Forest (configs/day06_baseline.json 설정)
# - 평가: Window 단위 정확도 + Clip 단위 다수결 정확도
#
# 출력: reports/motion_capture/
#   - summary.txt, summary.csv
#   - clip_predictions.csv
#   - confusion_<feature>_<test>.csv

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402
from src.sequence_utils import (  # noqa: E402
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
)

ACTION_FILE = ROOT / "configs" / "motion_capture_action_classes.csv"
CONFIG_PATH = ROOT / "configs" / "day06_baseline.json"
JOINT = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        default=None,
        help="쉼표로 구분 (기본: keypoint 폴더의 모든 Subject)",
    )
    parser.add_argument(
        "--keypoint-root",
        default=str(ROOT / "data" / "motion_capture_keypoints"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "motion_capture"),
    )
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=0.25)
    return parser.parse_args()


def load_actions() -> dict[str, str]:
    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        return {
            row["action_id"]: row["description"]
            for row in csv.DictReader(file)
        }


# ----------------------------------------------------
# robust Feature: 방향에 덜 민감한 Frame Feature 16개
# ----------------------------------------------------
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def robust_frame_features(df: pd.DataFrame) -> np.ndarray:
    points = np.stack(
        [
            df[[f"{n}_x_px", f"{n}_y_px"]].to_numpy(dtype=float)
            for n in COCO_KEYPOINT_NAMES
        ],
        axis=1,
    )

    def g(name):
        return points[:, JOINT[name]]

    shoulder = (g("left_shoulder") + g("right_shoulder")) / 2
    hip = (g("left_hip") + g("right_hip")) / 2
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))

    columns = []
    for side in ("left", "right"):
        columns += [
            _angle(g(f"{side}_shoulder"), g(f"{side}_elbow"), g(f"{side}_wrist")),
            _angle(g(f"{side}_hip"), g(f"{side}_knee"), g(f"{side}_ankle")),
            _angle(g(f"{side}_elbow"), g(f"{side}_shoulder"), g(f"{side}_hip")),
            _angle(g(f"{side}_shoulder"), g(f"{side}_hip"), g(f"{side}_knee")),
        ]
    trunk = shoulder - hip
    columns.append(np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1])))
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"):
        columns.append((g(name)[:, 1] - hip[:, 1]) / torso)
    columns.append(hip[:, 1] / torso)

    features = np.stack(columns, axis=1)
    # 좌우 쌍을 (작은 값, 큰 값)으로 정렬 → 좌우 반전에 무관
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7), (9, 10), (11, 12)]:
        low = np.minimum(features[:, a], features[:, b])
        high = np.maximum(features[:, a], features[:, b])
        features[:, a], features[:, b] = low, high
    return features


def window_ranges(n_frames: int, fps: float, window_sec, stride_sec):
    window = int(round(window_sec * fps))
    stride = max(1, int(round(stride_sec * fps)))
    for start in range(0, max(1, n_frames - window + 1), stride):
        yield start, start + window


def clip_vectors(csv_path: Path, args, config) -> dict[str, list]:
    df = pd.read_csv(csv_path).sort_values("frame_id").reset_index(drop=True)
    fps = 1.0 / float(df["time_sec"].diff().median())
    steps = int(config["sequence_steps"])
    min_valid = int(config["min_valid_frames"])

    seq_features, seq_ids = build_frame_feature_matrix(df)
    robust = robust_frame_features(df)
    valid = df["pose_detected"].to_numpy() == 1

    out = {"seq": [], "robust": []}
    for start, end in window_ranges(len(df), fps, args.window_sec, args.stride_sec):
        mask = (seq_ids >= start) & (seq_ids < end)
        rows = robust[start:end][valid[start:end]]
        rows = rows[np.isfinite(rows).all(axis=1)]
        if mask.sum() < min_valid or len(rows) < min_valid:
            continue
        sequence = resample_sequence(
            seq_features[mask], seq_ids[mask], target_steps=steps,
        )
        out["seq"].append(flatten_sequence(sequence))
        velocity = np.abs(np.diff(rows, axis=0))
        out["robust"].append(
            np.r_[
                rows.mean(0), rows.std(0), rows.min(0), rows.max(0),
                velocity.mean(0),
            ]
        )
    return out


def build_dataset(args, config, subjects) -> pd.DataFrame:
    records = []
    for subject in subjects:
        for csv_path in sorted(
            (Path(args.keypoint_root) / subject).glob("*.csv")
        ):
            _, action_id, take = csv_path.stem.split("_")
            vectors = clip_vectors(csv_path, args, config)
            for seq_x, robust_x in zip(vectors["seq"], vectors["robust"]):
                records.append(
                    {
                        "clip_id": csv_path.stem,
                        "subject": subject,
                        "action_id": action_id,
                        "take": take,
                        "seq": seq_x,
                        "robust": robust_x,
                    }
                )
    return pd.DataFrame(records)


def evaluate(
    name: str,
    feature: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    actions: dict,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    # 학습 쪽에 없는 동작은 테스트에서 제외합니다.
    test = test[test["action_id"].isin(set(train["action_id"]))]

    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        random_state=int(config["random_state"]),
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(np.stack(train[feature]), train["action_id"])
    test = test.assign(pred=model.predict(np.stack(test[feature])))

    clip_rows = []
    for clip_id, group in test.groupby("clip_id"):
        top = Counter(group["pred"]).most_common(1)[0][0]
        truth = group["action_id"].iloc[0]
        clip_rows.append(
            {
                "test": name,
                "feature": feature,
                "clip_id": clip_id,
                "action_id": truth,
                "action": actions[truth],
                "predicted": top,
                "predicted_action": actions[top],
                "correct": int(top == truth),
                "windows": len(group),
                "window_acc": round(float((group["pred"] == truth).mean()), 3),
            }
        )

    labels = sorted(set(train["action_id"]))
    matrix = confusion_matrix(test["action_id"], test["pred"], labels=labels)
    safe_name = name.replace("->", "_to_").replace(" ", "")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / f"confusion_{feature}_{safe_name}.csv",
        encoding="utf-8-sig",
    )

    summary = {
        "test": name,
        "feature": feature,
        "classes": len(labels),
        "train_windows": len(train),
        "test_windows": len(test),
        "test_clips": len(clip_rows),
        "window_acc": round(accuracy_score(test["action_id"], test["pred"]), 3),
        "clip_acc": round(float(np.mean([r["correct"] for r in clip_rows])), 3),
    }
    return summary, clip_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = [s.strip().upper() for s in args.subjects.split(",")]
    else:
        subjects = sorted(
            p.name for p in Path(args.keypoint_root).iterdir()
            if p.is_dir() and p.name.startswith("S")
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    actions = load_actions()

    data = build_dataset(args, config, subjects)
    print(f"Subjects: {subjects}")
    print(data.groupby(["subject", "take"]).size().to_string())
    print()

    # 테스트 목록: (이름, 학습 조건, 테스트 조건)
    splits = []
    for train_take, test_take in [("T01", "T02"), ("T02", "T01")]:
        splits.append(
            (
                f"view {train_take}->{test_take}",
                data["take"] == train_take,
                data["take"] == test_take,
            )
        )
    if len(subjects) >= 2:
        for subject in subjects:
            splits.append(
                (
                    f"person ->{subject}",
                    data["subject"] != subject,
                    data["subject"] == subject,
                )
            )

    summaries, clip_rows = [], []
    for name, train_mask, test_mask in splits:
        for feature in ("seq", "robust"):
            summary, rows = evaluate(
                name, feature, data[train_mask], data[test_mask],
                config, actions, output_dir,
            )
            summaries.append(summary)
            clip_rows.extend(rows)

    pd.DataFrame(summaries).to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(clip_rows).to_csv(
        output_dir / "clip_predictions.csv", index=False, encoding="utf-8-sig",
    )

    lines = [
        f"Subjects: {', '.join(subjects)}",
        f"Window: {args.window_sec}s / stride {args.stride_sec}s",
        f"Chance: {1 / len(actions):.3f} ({len(actions)} actions)",
        "",
        f"{'test':18} {'feature':8} {'clips':>5} {'window_acc':>10} {'clip_acc':>8}",
    ]
    for s in summaries:
        lines.append(
            f"{s['test']:18} {s['feature']:8} {s['test_clips']:>5} "
            f"{s['window_acc']:>10} {s['clip_acc']:>8}"
        )
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
