from __future__ import annotations

# ====================================================
# 48번. Motion Capture 17개 동작 학습·테스트
# ====================================================
# 촬영 방향: T01=앞, T02=오른쪽, T03=뒤
#            T02M=T02를 좌우 반전한 "왼쪽" 증강 데이터 (49번 결과, 학습 전용)
#
# 테스트 종류
#   1) 사람 교차 (Leave-One-Subject-Out)
#      한 사람을 빼고 나머지로 학습 → 뺀 사람으로 테스트. 사람 수만큼 반복.
#      처음 보는 사람에게 얼마나 통하는지 = 실제 성능에 가장 가까운 값
#   2) 방향 교차 (Leave-One-View-Out)
#      한 방향을 빼고 나머지 방향으로 학습 → 뺀 방향으로 테스트.
#      (T02를 테스트할 때는 T02에서 만든 T02M도 학습에서 뺍니다)
#
# 비교 조건: Feature(seq / robust) × 증강(없음 / T02M 추가)
#   seq    : day06 Frame Feature 38개(어깨너비 정규화 좌표 + 팔꿈치/무릎 각도)를
#            30 Step으로 Resample해 펼친 값
#   robust : 방향에 덜 민감한 Frame Feature 16개(관절 각도, 몸통 기울기,
#            엉덩이 기준 손·발·코 높이)의 창 안 통계. 좌우 쌍은 작은 값/큰 값으로
#            정렬하여 좌우 반전에 영향받지 않습니다.
#   robust2: robust 16개 + 팔·다리 뻗은 거리, 손·발 속도 등 20개 = 36개의 창 안 통계.
#            던지기·샷건·앞차기처럼 팔다리를 빠르게 뻗는 동작을 더 잘 구분합니다.
#   hybrid : robust2 창 모델 + 영상 전체 요약을 보는 Clip 모델의 확률을 더해 판정.
#            눕기/넘어지기/쪼그려 빼내기처럼 "어떻게 바닥까지 갔는지"가 중요한 동작용.
#            (영상의 시작·끝이 정해져 있어야 하므로 영상 단위 판정에만 쓸 수 있음)
#
# - Sliding Window: 2초 창을 0.25초 간격으로 자릅니다.
# - 모델: Random Forest (configs/day06_baseline.json 설정)
# - 평가: Window 정확도 + Clip(영상) 단위 다수결 정확도
# - 사람 교차 평균이 가장 높은 조건으로 전체 데이터를 학습해 모델을 저장합니다.
#
# 출력
#   reports/motion_capture/summary.txt, summary.csv, clip_predictions.csv,
#                          confusion_<조건>.csv (사람 교차 전체 합산)
#   artifacts/motion_capture/action_model.joblib, model_info.json

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

VIEW_NAMES = {
    "T01": "front",
    "T02": "right",
    "T03": "back",
    "T02M": "left(mirror)",
}
REAL_VIEWS = ["T01", "T02", "T03"]
FEATURES = ("seq", "robust", "robust2")


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
        "--aug-root",
        default=str(ROOT / "data" / "motion_capture_keypoints_aug"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "motion_capture"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "artifacts" / "motion_capture"),
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


# ----------------------------------------------------
# robust2 추가 Frame Feature 20개 (robust 16개 뒤에 붙여 36개로 사용)
#   팔·다리 뻗은 거리, 손·발 사이 거리, 몸 세로/가로 비율,
#   손목·발목·코 속도(엉덩이 기준), 엉덩이 이동 속도. 모두 몸통 길이로 정규화.
#   → 던지기·샷건·앞차기처럼 "얼마나 빨리, 멀리 뻗는지"가 핵심인 동작 구분용
# ----------------------------------------------------
def extended_frame_features(df: pd.DataFrame) -> np.ndarray:
    points = np.stack(
        [
            df[[f"{n}_x_px", f"{n}_y_px"]].to_numpy(dtype=float)
            for n in COCO_KEYPOINT_NAMES
        ],
        axis=1,
    )

    def g(name):
        return points[:, JOINT[name]]

    def dist(a, b):
        return np.linalg.norm(a - b, axis=1) / torso

    def speed(p):
        return np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1)] / torso

    shoulder = (g("left_shoulder") + g("right_shoulder")) / 2
    hip = (g("left_hip") + g("right_hip")) / 2
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))

    columns = []
    for base, tip in [("shoulder", "wrist"), ("hip", "ankle"),
                      ("shoulder", "elbow"), ("hip", "knee")]:
        columns += [
            dist(g(f"left_{tip}"), g(f"left_{base}")),
            dist(g(f"right_{tip}"), g(f"right_{base}")),
        ]
    columns += [
        dist(g("left_wrist"), g("right_wrist")),
        dist(g("left_ankle"), g("right_ankle")),
        np.abs(g("left_ankle")[:, 1] - g("right_ankle")[:, 1]) / torso,
        np.abs(g("left_wrist")[:, 1] - g("right_wrist")[:, 1]) / torso,
        dist(g("nose"), hip),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 전부 NaN인 프레임
        width = np.nanmax(points[:, :, 0], 1) - np.nanmin(points[:, :, 0], 1)
        height = np.nanmax(points[:, :, 1], 1) - np.nanmin(points[:, :, 1], 1)
    columns.append(height / (width + 1e-6))
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"):
        columns.append(speed(g(name) - hip))
    columns.append(speed(hip))

    features = np.stack(columns, axis=1)
    # 좌우 쌍 정렬 (팔/다리 뻗은 거리 4쌍, 손목 속도, 발목 속도)
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (14, 15), (16, 17)]:
        low = np.minimum(features[:, a], features[:, b])
        high = np.maximum(features[:, a], features[:, b])
        features[:, a], features[:, b] = low, high
    return features


def window_stats(rows: np.ndarray) -> np.ndarray:
    velocity = np.abs(np.diff(rows, axis=0))
    return np.r_[
        rows.mean(0), rows.std(0), rows.min(0), rows.max(0), velocity.mean(0),
    ]


# ----------------------------------------------------
# hybrid: 영상 전체 요약 Feature (Clip 모델용)
#   2초 창은 대부분 "누워 있는 끝 자세"만 보여서 눕기/넘어지기/쪼그려 빼내기가
#   헷갈립니다. 영상 전체를 10구간으로 나눈 평균 자세 흐름과, 엉덩이·코·어깨가
#   가장 빨리 내려가고/올라간 속도와 시점을 추가로 봅니다.
# ----------------------------------------------------
def vertical_speed_features(df: pd.DataFrame) -> np.ndarray:
    def center(a, b):
        return (
            df[[f"{a}_x_px", f"{a}_y_px"]].to_numpy(dtype=float)
            + df[[f"{b}_x_px", f"{b}_y_px"]].to_numpy(dtype=float)
        ) / 2

    shoulder = center("left_shoulder", "right_shoulder")
    hip = center("left_hip", "right_hip")
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))
    nose_y = df["nose_y_px"].to_numpy(dtype=float)
    # 화면 아래 방향(+)으로의 세로 속도, 몸통 길이/프레임
    return np.stack(
        [np.r_[0.0, np.diff(y)] / torso for y in (hip[:, 1], nose_y, shoulder[:, 1])],
        axis=1,
    )


def clip_descriptor(frames: np.ndarray, vertical: np.ndarray, parts=10) -> np.ndarray:
    data = np.c_[frames, vertical]
    timeline = [part.mean(0) for part in np.array_split(data, parts)]
    smooth = pd.DataFrame(vertical).rolling(5, min_periods=1).mean().to_numpy()
    return np.r_[
        np.concatenate(timeline),
        data.std(0), data.min(0), data.max(0),
        smooth.max(0), smooth.min(0), np.argmax(smooth, 0) / len(smooth),
    ]


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
    robust2 = np.c_[robust, extended_frame_features(df)]
    valid = df["pose_detected"].to_numpy() == 1

    def clean(matrix, start, end):
        rows = matrix[start:end][valid[start:end]]
        return rows[np.isfinite(rows).all(axis=1)]

    out = {"seq": [], "robust": [], "robust2": []}
    for start, end in window_ranges(len(df), fps, args.window_sec, args.stride_sec):
        mask = (seq_ids >= start) & (seq_ids < end)
        rows = clean(robust, start, end)
        rows2 = clean(robust2, start, end)
        if mask.sum() < min_valid or min(len(rows), len(rows2)) < min_valid:
            continue
        sequence = resample_sequence(
            seq_features[mask], seq_ids[mask], target_steps=steps,
        )
        out["seq"].append(flatten_sequence(sequence))
        out["robust"].append(window_stats(rows))
        out["robust2"].append(window_stats(rows2))

    vertical = vertical_speed_features(df)
    whole = valid & np.isfinite(robust2).all(axis=1) & np.isfinite(vertical).all(axis=1)
    out["clip"] = (
        clip_descriptor(robust2[whole], vertical[whole])
        if whole.sum() >= min_valid else None
    )
    return out


def build_dataset(args, config, subjects) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(창 단위 데이터, 영상 단위 요약 데이터)"""
    records, clip_records = [], []
    roots = [(Path(args.keypoint_root), False)]
    if Path(args.aug_root).exists():
        roots.append((Path(args.aug_root), True))

    for root, is_aug in roots:
        for subject in subjects:
            for csv_path in sorted((root / subject).glob("*.csv")):
                _, action_id, take = csv_path.stem.split("_")
                vectors = clip_vectors(csv_path, args, config)
                if vectors["clip"] is not None:
                    clip_records.append(
                        {
                            "clip_id": csv_path.stem,
                            "action_id": action_id,
                            "clip": vectors["clip"],
                        }
                    )
                for i in range(len(vectors["seq"])):
                    records.append(
                        {
                            "clip_id": csv_path.stem,
                            "subject": subject,
                            "action_id": action_id,
                            "take": take,
                            "aug": is_aug,
                            **{key: vectors[key][i] for key in FEATURES},
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(clip_records).set_index("clip_id")


def make_model(config: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        random_state=int(config["random_state"]),
        class_weight="balanced",
        n_jobs=-1,
    )


def fit_models(train: pd.DataFrame, feature: str, config: dict, clip_data):
    """창 모델(과 hybrid이면 영상 모델)을 학습해 (window_model, clip_model)로 반환."""
    window_feature = "robust2" if feature == "hybrid" else feature
    window_model = make_model(config)
    window_model.fit(np.stack(train[window_feature]), train["action_id"])
    if feature != "hybrid":
        return window_model, None
    clips = clip_data.loc[clip_data.index.intersection(train["clip_id"].unique())]
    clip_model = make_model(config)
    clip_model.fit(np.stack(list(clips["clip"])), clips["action_id"])
    return window_model, clip_model


def run_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    config: dict,
    clip_data: pd.DataFrame,
) -> pd.DataFrame:
    window_model, clip_model = fit_models(train, feature, config, clip_data)
    window_feature = "robust2" if feature == "hybrid" else feature
    proba = window_model.predict_proba(np.stack(test[window_feature]))
    classes = window_model.classes_
    out = test.assign(pred=classes[proba.argmax(1)])
    if clip_model is None:
        return out

    # hybrid 판정 = (창 확률 평균 + 영상 모델 확률)이 가장 큰 동작
    assert list(clip_model.classes_) == list(classes)
    window_mean = pd.DataFrame(proba, index=test.index).groupby(test["clip_id"]).mean()
    clip_ids = window_mean.index.intersection(clip_data.index)
    combined = window_mean.loc[clip_ids].to_numpy() + clip_model.predict_proba(
        np.stack(list(clip_data.loc[clip_ids, "clip"]))
    )
    clip_pred = dict(zip(clip_ids, classes[combined.argmax(1)]))
    return out.assign(clip_pred=out["clip_id"].map(clip_pred))


def clip_votes(predicted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clip_id, group in predicted.groupby("clip_id"):
        truth = group["action_id"].iloc[0]
        if "clip_pred" in group and pd.notna(group["clip_pred"].iloc[0]):
            top = group["clip_pred"].iloc[0]
        else:
            top = Counter(group["pred"]).most_common(1)[0][0]
        rows.append(
            {
                "clip_id": clip_id,
                "subject": group["subject"].iloc[0],
                "take": group["take"].iloc[0],
                "view": VIEW_NAMES[group["take"].iloc[0]],
                "action_id": truth,
                "predicted": top,
                "correct": int(top == truth),
                "windows": len(group),
                "window_acc": round(float((group["pred"] == truth).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("confusion_*.csv"):
        old.unlink()

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
    labels = sorted(actions)

    data, clip_data = build_dataset(args, config, subjects)
    real = data[~data["aug"]]
    has_aug = bool(data["aug"].any())
    print(f"Subjects: {subjects}")
    print(
        data.groupby(["take"])["clip_id"].nunique()
        .rename("clips").to_string()
    )
    print(f"Windows: real={len(real)}, aug={int(data['aug'].sum())}")
    print()

    conditions = [
        (feature, aug)
        for feature in (*FEATURES, "hybrid")
        for aug in ((False, True) if has_aug else (False,))
    ]

    summaries, clip_tables = [], []
    for feature, use_aug in conditions:
        cond = f"{feature}{'+aug' if use_aug else ''}"
        pool = data if use_aug else real

        # 1) 사람 교차
        person_preds = []
        for subject in subjects:
            train = pool[pool["subject"] != subject]
            test = real[real["subject"] == subject]
            person_preds.append(run_fold(train, test, feature, config, clip_data))
        person_preds = pd.concat(person_preds)
        clips = clip_votes(person_preds).assign(test="person", condition=cond)
        clip_tables.append(clips)

        matrix = confusion_matrix(
            person_preds["action_id"], person_preds["pred"], labels=labels,
        )
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output_dir / f"confusion_person_{cond}.csv",
            encoding="utf-8-sig",
        )

        row = {
            "condition": cond,
            "person_window_acc": round(
                float((person_preds["pred"] == person_preds["action_id"]).mean()), 3,
            ),
            "person_clip_acc": round(float(clips["correct"].mean()), 3),
        }
        for subject in subjects:
            row[f"person_{subject}"] = round(
                float(clips[clips["subject"] == subject]["correct"].mean()), 3,
            )
        for take in REAL_VIEWS:
            row[f"person_view_{VIEW_NAMES[take]}"] = round(
                float(clips[clips["take"] == take]["correct"].mean()), 3,
            )

        # 2) 방향 교차
        for take in REAL_VIEWS:
            excluded = {take, f"{take}M"}
            train = pool[~pool["take"].isin(excluded)]
            test = real[real["take"] == take]
            view_clips = clip_votes(
                run_fold(train, test, feature, config, clip_data)
            )
            clip_tables.append(
                view_clips.assign(test=f"view->{take}", condition=cond)
            )
            row[f"view_{VIEW_NAMES[take]}"] = round(
                float(view_clips["correct"].mean()), 3,
            )

        summaries.append(row)
        print(
            f"{cond:12} person_clip_acc={row['person_clip_acc']} "
            f"view(front/right/back)="
            f"{row['view_front']}/{row['view_right']}/{row['view_back']}"
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    clip_table = pd.concat(clip_tables)
    clip_table["action"] = clip_table["action_id"].map(actions)
    clip_table["predicted_action"] = clip_table["predicted"].map(actions)
    clip_table.to_csv(
        output_dir / "clip_predictions.csv", index=False, encoding="utf-8-sig",
    )

    # 사람 교차 평균이 가장 높은 조건으로 최종 모델 학습·저장
    best = summary.sort_values("person_clip_acc", ascending=False).iloc[0]
    best_feature = best["condition"].split("+")[0]
    best_pool = data if best["condition"].endswith("+aug") else real
    final_model, final_clip_model = fit_models(
        best_pool, best_feature, config, clip_data,
    )
    joblib.dump(final_model, model_dir / "action_model.joblib")
    clip_model_path = model_dir / "clip_model.joblib"
    if final_clip_model is not None:
        joblib.dump(final_clip_model, clip_model_path)
    elif clip_model_path.exists():
        clip_model_path.unlink()
    (model_dir / "model_info.json").write_text(
        json.dumps(
            {
                "condition": best["condition"],
                "feature": best_feature,
                "window_feature": "robust2" if best_feature == "hybrid" else best_feature,
                "clip_model": (
                    "clip_model.joblib" if final_clip_model is not None else None
                ),
                "window_sec": args.window_sec,
                "stride_sec": args.stride_sec,
                "subjects": subjects,
                "train_windows": len(best_pool),
                "classes": list(final_model.classes_),
                "person_clip_acc": float(best["person_clip_acc"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 동작별 정확도 (최종 조건, 사람 교차)
    best_clips = clip_table[
        (clip_table["test"] == "person")
        & (clip_table["condition"] == best["condition"])
    ]
    per_action = (
        best_clips.groupby("action_id")
        .agg(correct=("correct", "sum"), clips=("correct", "size"))
    )
    top_confusion = (
        best_clips[best_clips["correct"] == 0]
        .groupby("action_id")["predicted"]
        .agg(lambda s: s.value_counts().index[0])
    )

    lines = [
        f"Subjects: {', '.join(subjects)}",
        "Views: T01=front, T02=right, T03=back, T02M=left(mirror, train only)",
        f"Window: {args.window_sec}s / stride {args.stride_sec}s",
        f"Chance: {1 / len(actions):.3f} ({len(actions)} actions)",
        "",
        "Clip accuracy",
        summary.to_string(index=False),
        "",
        f"Final model: {best['condition']} -> {model_dir / 'action_model.joblib'}",
        "",
        f"Per action (person test, {best['condition']})",
    ]
    for action_id, item in per_action.iterrows():
        wrong = top_confusion.get(action_id)
        note = f"  (자주 틀림 → {actions[wrong]})" if wrong else ""
        lines.append(
            f"{action_id} {actions[action_id]:12} "
            f"{item['correct']}/{item['clips']}{note}"
        )
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)


if __name__ == "__main__":
    main()
