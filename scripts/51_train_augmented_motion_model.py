from __future__ import annotations

# ====================================================
# 51번. 관절 좌표 증강으로 hybrid 모델 재학습·비교 (영상 재촬영 없음)
# ====================================================
# 48번 hybrid 모델(robust2 창 모델 + Clip 모델)은 60fps·고정 카메라·동작만 담긴 클립으로만
# 학습되어, 웹 서비스의 IP Webcam(낮은 FPS, 8초 수집의 정지 구간, 손 떨림)에서 신뢰도가 낮습니다.
# 46번이 뽑아 둔 관절 CSV를 변형해 학습 데이터를 늘리고, 효과가 있는 증강만 고릅니다.
#
# 증강 (관절 트랙 단위)
#   fps    : 10~30fps ±20% 간격으로 다시 뽑기 (휴대폰·웹캠 FPS)
#   pad    : 동작 앞뒤에 0.3~3초 정지(미세 흔들림) 구간 (정해진 시간 수집)
#   mirror : 좌우 반전 + 왼쪽/오른쪽 관절 교환 (모든 방향)
#   speed  : 시간축 0.8~1.25배 (사람마다 다른 동작 속도)
#   camera : ±8° 기울기, 0.85~1.15배 크기, 느린 흔들림 (손에 든 카메라)
#   noise  : 관절 위치 잡음, 프레임 일부 누락, 손목·발목 잠깐 미검출
#
# 학습·추론 경로는 웹 서비스(ai_service)와 같은 코드를 씁니다.
#   - 트랙을 학습 FPS(59.88)로 보간: ai_service/services/pose_pipeline.resample_track
#   - 정지 구간 축소 + 기하평균: ai_service/services/temporal.trim_static
#
# 평가: 사람 교차 검증(5명 중 1명을 빼고 학습 → 뺀 사람으로 시험). 시험 데이터는 증강하지 않은
#       실제 영상이고, 조건 3가지로 시험합니다.
#   clean  : 원본 그대로 (기존 성능 유지 확인)
#   wc15   : 웹캠 재현 — 8초 수집(앞뒤 정지) + 15fps ±20% + 관절 흔들림
#   wc60   : 웹캠 재현 — 8초 수집(앞뒤 정지) + 60fps + 관절 흔들림 (이 프로젝트 휴대폰 조건)
#
# 사용
#   .venv/bin/python scripts/51_train_augmented_motion_model.py --experiments all
#   .venv/bin/python scripts/51_train_augmented_motion_model.py --final mix8   # 최종 모델 저장

import argparse
import copy
import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVICE = Path("/mnt/d/Study_data/교과10/교과10 최종프로젝트/vision_action_platform/ai_service")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--service-dir", default=str(DEFAULT_SERVICE), help="웹 서비스 ai_service 폴더")
    p.add_argument("--keypoint-root", default=str(ROOT / "data" / "motion_capture_keypoints"))
    p.add_argument("--output-dir", default=str(ROOT / "reports" / "motion_capture_aug"))
    p.add_argument("--model-dir", default=str(ROOT / "artifacts" / "motion_capture_aug"))
    p.add_argument("--experiments", default="all", help="쉼표 구분 실험 이름 또는 all")
    p.add_argument("--final", default=None, help="이 실험 설정으로 전체 5명 학습 후 모델 저장")
    p.add_argument("--oof", default=None,
                   help="이 실험 설정으로 사람 교차 검증을 돌려 처음 보는 사람의 예측 확률을 CSV로 저장 (신뢰도 보정용)")
    p.add_argument("--test-seeds", type=int, default=2, help="웹캠 재현 시험 반복 수")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--raw48", action="store_true", help="원본 클립은 48번처럼 CSV 그대로 계산해 학습")
    return p.parse_args()


ARGS = parse_args()
sys.path.insert(0, str(Path(ARGS.service_dir)))  # 서비스와 같은 src·services 모듈 사용

from services.pose_pipeline import TrackData, resample_track  # noqa: E402
from services.temporal import trim_static  # noqa: E402
from src.motion_capture_features import (  # noqa: E402
    clip_descriptor,
    extended_frame_features,
    robust_frame_features,
    valid_keypoints,
    vertical_speed_features,
    window_stats,
)

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
SERVICE = json.loads((Path(ARGS.service_dir) / "models" / "model_info.json").read_text(encoding="utf-8"))
PROJECT = json.loads((Path(ARGS.service_dir) / "project.json").read_text(encoding="utf-8"))
FPS = float(SERVICE["fps"])                      # 59.88
WINDOW = int(round(float(SERVICE["window_sec"]) * FPS))
STRIDE = max(1, int(round(float(SERVICE["stride_sec"]) * FPS)))
MIN_VALID = int(SERVICE["min_valid_frames"])
MAX_GAP = float(PROJECT.get("max_gap_seconds", 0.25))
TRIM = PROJECT.get("static_trim", {})
TRIM_KW = {"pre": TRIM.get("pre_sec", 0.6), "post": TRIM.get("post_sec", 0.8),
           "motion_rel": TRIM.get("motion_rel", 0.2), "min_frames": MIN_VALID}
UNCERTAIN = float(PROJECT.get("uncertain_threshold", 0.4))
RF = {"n_estimators": 300, "max_depth": None, "random_state": 42}


# ----------------------------------------------------
# 데이터: 46번 관절 CSV → 서비스와 같은 트랙(핵심 관절이 보이는 프레임만)
# ----------------------------------------------------
def load_track(path: Path) -> dict | None:
    df = pd.read_csv(path)
    times, points = [], []
    for _, row in df[df["pose_detected"] == 1].iterrows():
        xy = np.array([[row[f"{j}_x_px"], row[f"{j}_y_px"]] for j in JOINTS], dtype=np.float32)
        xy = valid_keypoints(np.nan_to_num(xy))
        if xy is not None:
            times.append(float(row["time_sec"]))
            points.append(xy)
    if len(times) < MIN_VALID:
        return None
    subject, action, take = path.stem.split("_")
    return {"subject": subject, "action": action, "take": take, "name": path.stem,
            "times": np.array(times) - times[0], "points": np.stack(points)}


def raw48_vectors(item):
    """48번 학습과 같은 계산: 60fps CSV 그대로(보간 없음), pose_detected 프레임으로 창·Clip Feature."""
    df = pd.read_csv(Path(ARGS.keypoint_root) / item["subject"] / f"{item['name']}.csv")
    df = df.sort_values("frame_id").reset_index(drop=True)
    fps = 1.0 / float(df["time_sec"].diff().median())
    robust2 = np.c_[robust_frame_features(df), extended_frame_features(df)]
    valid = df["pose_detected"].to_numpy() == 1
    window, stride = int(round(2.0 * fps)), max(1, int(round(0.25 * fps)))
    windows = []
    for start in range(0, max(1, len(df) - window + 1), stride):
        rows = robust2[start:start + window][valid[start:start + window]]
        rows = rows[np.isfinite(rows).all(axis=1)]
        if len(rows) >= MIN_VALID:
            windows.append(window_stats(rows))
    vertical = vertical_speed_features(df)
    whole = valid & np.isfinite(robust2).all(axis=1) & np.isfinite(vertical).all(axis=1)
    clip = clip_descriptor(robust2[whole], vertical[whole]) if whole.sum() >= MIN_VALID else None
    return windows, clip


def train_vectors(item):
    """원본 클립은 RAW48 옵션이면 48번 경로, 증강본은 서비스 경로."""
    if RAW48 and "#aug" not in item["name"]:
        return raw48_vectors(item)
    return track_vectors(item)


RAW48 = ARGS.raw48  # 작업 프로세스에서도 같은 값을 쓰도록 모듈에서 설정


def to_track(item) -> TrackData:
    t = TrackData(1)
    t.times = list(map(float, item["times"]))
    t.keypoints = list(item["points"].astype(np.float32))
    t.confidences = [0.9] * len(t.times)
    t.frame_indices = list(range(len(t.times)))
    return t


# ----------------------------------------------------
# Feature (ai_service action_inference.track_vectors와 같은 계산)
# ----------------------------------------------------
def track_vectors(item):
    track = to_track(item)
    if len(track.times) < MIN_VALID:
        return [], None
    df = resample_track(track, FPS, MAX_GAP)
    frame = np.c_[robust_frame_features(df), extended_frame_features(df)]
    valid = np.isfinite(frame).all(axis=1)
    windows = []
    n = len(df)
    for start in range(0, max(1, n - WINDOW + 1), STRIDE):
        rows = frame[start:min(n, start + WINDOW)][valid[start:min(n, start + WINDOW)]]
        if len(rows) >= MIN_VALID:
            windows.append(window_stats(rows))
    vertical = vertical_speed_features(df)
    whole = valid & np.isfinite(vertical).all(axis=1)
    clip = clip_descriptor(frame[whole], vertical[whole]) if whole.sum() >= MIN_VALID else None
    return windows, clip


# ----------------------------------------------------
# 증강
# ----------------------------------------------------
def _torso(p):
    return float(np.nanmedian(np.linalg.norm(p[:, [5, 6]].mean(1) - p[:, [11, 12]].mean(1), axis=1)))


def _keep_zero(new, old):
    """YOLO가 못 찾은 관절(0,0)은 변형 후에도 (0,0)으로 둠."""
    missing = np.all(old == 0, axis=-1)
    new[missing] = 0
    return new


def aug_fps(item, rng):
    fps = rng.uniform(10, 30)
    t = item["times"]
    picked, cur = [], t[0]
    while cur <= t[-1]:
        picked.append(cur)
        cur += (1 / fps) * rng.uniform(0.8, 1.2)
    idx = np.unique(np.searchsorted(t, picked).clip(0, len(t) - 1))
    return {**item, "times": t[idx], "points": item["points"][idx]}


def aug_pad(item, rng, lo=0.3, hi=3.0):
    t, p = item["times"], item["points"]
    torso = _torso(p)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 60

    def still(base, sec):
        n = max(1, int(sec / dt))
        sway = rng.normal(0, 0.004 * torso, (n, 17, 2)).cumsum(0) * 0.15
        return _keep_zero(base + sway, np.repeat(base[None], n, 0)), n

    pre_p, n_pre = still(p[0], rng.uniform(lo, hi))
    post_p, n_post = still(p[-1], rng.uniform(lo, hi))
    times = np.r_[np.arange(n_pre) * dt, n_pre * dt + t, n_pre * dt + t[-1] + dt * (1 + np.arange(n_post))]
    return {**item, "times": times, "points": np.concatenate([pre_p, p, post_p]).astype(np.float32)}


def aug_mirror(item, rng=None):
    p = item["points"].copy()
    center = np.nanmedian(p[:, [11, 12], 0][p[:, [11, 12], 0] > 0])
    q = p.copy()
    q[..., 0] = 2 * center - p[..., 0]
    for a, b in LR_PAIRS:
        q[:, [a, b]] = q[:, [b, a]]
    swapped_old = p.copy()
    for a, b in LR_PAIRS:
        swapped_old[:, [a, b]] = swapped_old[:, [b, a]]
    return {**item, "points": _keep_zero(q, swapped_old).astype(np.float32)}


def aug_speed(item, rng):
    return {**item, "times": item["times"] * rng.uniform(0.8, 1.25)}


def aug_camera(item, rng):
    p = item["points"].astype(float)
    t = item["times"]
    torso = _torso(p)
    center = np.nanmedian(p[p.sum(-1) > 0], axis=0)
    angle = np.deg2rad(rng.uniform(-8, 8))
    scale = rng.uniform(0.85, 1.15)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * scale
    q = (p - center) @ rot.T + center
    # 손에 든 카메라: 0.3~1.5Hz의 느린 흔들림 (몸통 길이의 0~12%)
    amp = rng.uniform(0, 0.12) * torso
    f1, f2 = rng.uniform(0.3, 1.5, 2)
    ph = rng.uniform(0, 2 * np.pi, 2)
    shake = np.stack([np.sin(2 * np.pi * f1 * t + ph[0]), np.sin(2 * np.pi * f2 * t + ph[1])], 1) * amp
    q = q + shake[:, None, :]
    return {**item, "points": _keep_zero(q, p).astype(np.float32)}


def aug_noise(item, rng):
    p = item["points"].astype(float)
    t = item["times"]
    torso = _torso(p)
    q = p + rng.normal(0, rng.uniform(0.003, 0.012) * torso, p.shape)
    q = _keep_zero(q, p)
    # 손목·발목이 잠깐(0.1~0.4초) 안 보이는 구간 0~2개
    for _ in range(rng.integers(0, 3)):
        joint = rng.choice([9, 10, 15, 16])
        start = rng.uniform(t[0], t[-1])
        q[(t >= start) & (t <= start + rng.uniform(0.1, 0.4)), joint] = 0
    keep = rng.random(len(t)) > rng.uniform(0.0, 0.12)  # 프레임 0~12% 누락
    keep[[0, -1]] = True
    return {**item, "times": t[keep], "points": q[keep].astype(np.float32)}


AUGS = {"fps": aug_fps, "pad": aug_pad, "mirror": aug_mirror, "speed": aug_speed,
        "camera": aug_camera, "noise": aug_noise}
# 섞어 쓸 때 적용 순서 (실제 과정 순서: 사람 동작 → 촬영 → 카메라 수집 → 검출)
ORDER = ["mirror", "speed", "pad", "camera", "noise", "fps"]


def augment(item, names, rng, prob=1.0):
    out = item
    for name in ORDER:
        if name in names and rng.random() < prob:
            out = AUGS[name](out, rng)
    return out


# ----------------------------------------------------
# 실험 설정: (사용할 증강, 원본 1개당 증강본 수, 각 증강 적용 확률)
# ----------------------------------------------------
EXPERIMENTS = {
    "base": ([], 0, 1.0),
    "fps": (["fps"], 2, 1.0),
    "pad": (["pad"], 2, 1.0),
    "mirror": (["mirror"], 1, 1.0),
    "speed": (["speed"], 2, 1.0),
    "camera": (["camera"], 2, 1.0),
    "noise": (["noise"], 2, 1.0),
}


def mix(names, copies, prob=0.5):
    return (names, copies, prob)


# ----------------------------------------------------
# 학습·예측
# ----------------------------------------------------
def build_train(items, names, copies, prob, seed):
    """원본 + 증강본의 (트랙 목록). 증강본은 원본마다 copies개."""
    tracks = list(items)
    rng = np.random.default_rng(seed)
    for item in items:
        for c in range(copies):
            sub = np.random.default_rng(rng.integers(1 << 31))
            aug = augment(item, names, sub, prob)
            # 섞어 쓰기에서 아무 증강도 안 걸린 경우 하나는 꼭 적용
            if aug is item and names:
                aug = AUGS[sub.choice(names)](item, sub)
            tracks.append({**aug, "name": f"{item['name']}#aug{c}"})
    return tracks


def featurize(tracks, pool, train=False):
    return pool.map(train_vectors if train else track_vectors, tracks, chunksize=4)


def fit(tracks, feats):
    xw, yw, xc, yc = [], [], [], []
    for item, (windows, clip) in zip(tracks, feats):
        xw.extend(windows)
        yw.extend([item["action"]] * len(windows))
        if clip is not None:
            xc.append(clip)
            yc.append(item["action"])
    wm = RandomForestClassifier(**RF, class_weight="balanced", n_jobs=-1).fit(np.stack(xw), yw)
    cm = RandomForestClassifier(**RF, class_weight="balanced", n_jobs=-1).fit(np.stack(xc), yc)
    assert list(wm.classes_) == list(cm.classes_)
    return wm, cm


def proba_from(models, feats):
    wm, cm = models
    windows, clip = feats
    if not windows or clip is None:
        return None
    return (wm.predict_proba(np.stack(windows)).mean(0) + cm.predict_proba(clip.reshape(1, -1))[0]) / 2


def webcam(item, rng, fps):
    """시험용 웹캠 재현 (eval_webcam_sim.py와 같은 방식): 8초 수집 + fps ±20% + 관절 흔들림."""
    t = item["times"]
    p = item["points"].astype(float)
    torso = _torso(p)
    span = t[-1]
    pre = max(0.5, min(2.2, (8.0 - span) / 2))
    post = max(0.5, 8.0 - span - pre)
    dt = 1 / 60
    t_pre, t_post = np.arange(0, pre, dt), np.arange(0, post, dt)
    sway = lambda n, base: base + rng.normal(0, 0.004 * torso, (n, 17, 2)).cumsum(0) * 0.15
    full_t = np.r_[t_pre, pre + t, pre + span + dt + t_post]
    full_p = np.r_[sway(len(t_pre), p[0]), p, sway(len(t_post), p[-1])]
    if fps >= 59:
        idx = np.arange(len(full_t))
    else:
        picked, cur = [], 0.0
        while cur <= full_t[-1]:
            picked.append(cur)
            cur += (1 / fps) * rng.uniform(0.8, 1.2)
        idx = np.unique(np.searchsorted(full_t, picked).clip(0, len(full_t) - 1))
    q = full_p[idx] + rng.normal(0, 0.004 * torso, (len(idx), 17, 2))
    return {**item, "times": full_t[idx], "points": q.astype(np.float32)}


def test_variants(item):
    """한 시험 트랙에서 (전체 트랙, 정지 구간 축소 트랙)의 Feature — 서비스 추론과 같은 입력."""
    full = track_vectors(item)
    short, removed = trim_static(to_track(item), **TRIM_KW)
    if removed > 0:
        short_item = {**item, "times": np.array(short.times), "points": np.stack(short.keypoints)}
        return full, track_vectors(short_item)
    return full, None


def service_proba(models, pair, trim=True):
    """서비스와 같은 판정: 전체 확률, 정지 구간을 줄였다면 두 확률의 기하평균."""
    full, short = pair
    p = proba_from(models, full)
    if p is None:
        return None
    if trim and short is not None:
        q = proba_from(models, short)
        if q is not None:
            g = np.sqrt(p * q)
            if g.sum() > 0:
                p = g / g.sum()
    return p


def make_tests(items, seeds):
    tests = {"clean": [(it, it) for it in items]}
    for name, fps in (("wc15", 15.0), ("wc60", 60.0)):
        rows = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            rows.extend((it, webcam(it, np.random.default_rng(rng.integers(1 << 31)), fps)) for it in items)
        tests[name] = rows
    return tests


def score(models, test_rows, test_feats, trim=True):
    classes = models[0].classes_
    correct, confs, right_conf, wrong_conf = 0, [], [], []
    per_action = {}
    for (orig, _), pair in zip(test_rows, test_feats):
        p = service_proba(models, pair, trim)
        if p is None:
            pred, conf = None, 0.0
        else:
            pred, conf = classes[int(np.argmax(p))], float(np.max(p))
        ok = pred == orig["action"]
        correct += ok
        confs.append(conf)
        (right_conf if ok else wrong_conf).append(conf)
        a = per_action.setdefault(orig["action"], [0, 0])
        a[0] += ok
        a[1] += 1
    n = len(test_rows)
    return {
        "acc": correct / n,
        "conf": float(np.mean(confs)),
        "conf_right": float(np.mean(right_conf)) if right_conf else 0.0,
        "conf_wrong": float(np.mean(wrong_conf)) if wrong_conf else 0.0,
        "uncertain": float(np.mean(np.array(confs) < UNCERTAIN)),
        "per_action": per_action,
    }


# ----------------------------------------------------
def main():
    global RAW48
    RAW48 = ARGS.raw48
    out_dir, model_dir = Path(ARGS.output_dir), Path(ARGS.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    paths = sorted(p for p in Path(ARGS.keypoint_root).glob("S*/*.csv") if p.stem.split("_")[2] in ("T01", "T02", "T03"))
    with Pool(ARGS.workers) as pool:
        items = [it for it in pool.map(load_track, paths, chunksize=4) if it is not None]
        subjects = sorted({it["subject"] for it in items})
        print(f"트랙 {len(items)}개 (사람 {subjects}) 로드 {time.time() - started:.0f}s", flush=True)

        exps = dict(EXPERIMENTS)
        # 효과 확인 후 섞어 쓰는 실험 (아래 이름으로 --experiments에서 지정)
        exps.update({
            "mix4": mix(["fps", "pad", "mirror", "speed", "camera", "noise"], 4),
            "mix8": mix(["fps", "pad", "mirror", "speed", "camera", "noise"], 8),
        })

        if ARGS.final:
            names, copies, prob = parse_exp(ARGS.final, exps)
            print(f"최종 모델: {ARGS.final} = {names} ×{copies} p={prob}", flush=True)
            tracks = build_train(items, names, copies, prob, seed=7)
            feats = featurize(tracks, pool, train=True)
            models = fit(tracks, feats)
            save_models(models, model_dir, ARGS.final, len(tracks), subjects)
            print(f"저장: {model_dir}  ({time.time() - started:.0f}s)")
            return

        # 시험 데이터 Feature는 실험과 무관하게 한 번만 계산 (사람별)
        tests = make_tests(items, ARGS.test_seeds)
        if ARGS.oof:
            save_oof(ARGS.oof, exps, items, subjects, tests, pool, out_dir)
            return
        test_feats = {name: list(pool.map(test_variants, [row[1] for row in rows], chunksize=4))
                      for name, rows in tests.items()}
        print(f"시험 Feature 준비 {time.time() - started:.0f}s", flush=True)

        chosen = list(exps) if ARGS.experiments == "all" else ARGS.experiments.split(",")
        results = []
        for exp in chosen:
            names, copies, prob = parse_exp(exp, exps)
            t0 = time.time()
            # 조건마다 '정지 구간 축소+기하평균 적용(서비스 현재)'과 '미적용(NT)'을 함께 기록
            fold_scores = {f"{name}{sfx}": [] for name in tests for sfx in ("", "NT")}
            for subject in subjects:
                train_items = [it for it in items if it["subject"] != subject]
                tracks = build_train(train_items, names, copies, prob, seed=zlib.crc32(f"{exp}/{subject}".encode()))
                models = fit(tracks, featurize(tracks, pool, train=True))
                for name, rows in tests.items():
                    idx = [i for i, (orig, _) in enumerate(rows) if orig["subject"] == subject]
                    for sfx, trim in (("", True), ("NT", False)):
                        s = score(models, [rows[i] for i in idx], [test_feats[name][i] for i in idx], trim=trim)
                        fold_scores[f"{name}{sfx}"].append((s, len(idx)))
            row = {"experiment": exp, "augs": "+".join(names) or "-", "copies": copies, "prob": prob}
            for name, items_scores in fold_scores.items():
                total = sum(n for _, n in items_scores)
                for key in ("acc", "conf", "conf_right", "conf_wrong", "uncertain"):
                    row[f"{name}_{key}"] = round(sum(s[key] * n for s, n in items_scores) / total, 4)
                per_action = {}
                for s, _ in items_scores:
                    for a, (ok, n) in s["per_action"].items():
                        cur = per_action.setdefault(a, [0, 0])
                        cur[0] += ok
                        cur[1] += n
                row[f"{name}_per_action"] = json.dumps({a: f"{ok}/{n}" for a, (ok, n) in sorted(per_action.items())})
            results.append(row)
            line = "  ".join(
                f"{n} {row[n + '_acc']:.3f}/{row[n + '_conf']:.2f}/u{row[n + '_uncertain']:.2f}"
                for n in ("clean", "cleanNT", "wc15", "wc15NT", "wc60", "wc60NT")
            )
            print(f"{exp:28s} {line}  ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(results).to_csv(out_dir / "experiments.csv", index=False, encoding="utf-8-sig")
    print(f"완료 {time.time() - started:.0f}s → {out_dir / 'experiments.csv'}")


def save_oof(exp, exps, items, subjects, tests, pool, out_dir):
    """사람 교차 검증의 시험 예측(정지 구간 축소 없이, 현재 서비스 설정)을 행 단위로 저장."""
    names, copies, prob = parse_exp(exp, exps)
    feats = {name: list(pool.map(test_variants, [row[1] for row in rows], chunksize=4)) for name, rows in tests.items()}
    rows_out = []
    for subject in subjects:
        t0 = time.time()
        train_items = [it for it in items if it["subject"] != subject]
        tracks = build_train(train_items, names, copies, prob, seed=zlib.crc32(f"{exp}/{subject}".encode()))
        models = fit(tracks, featurize(tracks, pool, train=True))
        classes = list(models[0].classes_)
        for name, rows in tests.items():
            for i, (orig, _) in enumerate(rows):
                if orig["subject"] != subject:
                    continue
                p = service_proba(models, feats[name][i], trim=False)
                if p is None:
                    continue
                rows_out.append({"condition": name, "subject": subject, "clip": orig["name"],
                                 "action": orig["action"], **{c: float(v) for c, v in zip(classes, p)}})
        print(f"  {subject} 완료 ({time.time() - t0:.0f}s)", flush=True)
    path = out_dir / "oof_predictions.csv"
    pd.DataFrame(rows_out).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(rows_out)}행)")


def parse_exp(exp, exps):
    """이름 또는 '증강+증강:개수:확률' 형식 (예: fps+pad+camera:6:0.5)."""
    if exp in exps:
        return exps[exp]
    spec, copies, prob = (exp.split(":") + ["4", "0.5"])[:3]
    return spec.split("+"), int(copies), float(prob)


def save_models(models, model_dir, exp, n_tracks, subjects):
    wm, cm = models
    # 증강으로 트리가 커져 compress=3이면 47MB → xz로 24MB (예측 동일, 로드 약 1.3초)
    joblib.dump(wm, model_dir / "action_model.joblib", compress=("xz", 9))
    joblib.dump(cm, model_dir / "clip_model.joblib", compress=("xz", 9))
    info = dict(SERVICE)
    info.update({
        "condition": f"hybrid+aug({exp})",
        "augmentation": exp,
        "train_tracks": n_tracks,
        "subjects": subjects,
        "classes": list(wm.classes_),
        "trained_with": "scripts/51_train_augmented_motion_model.py (서비스와 같은 59.88fps 보간 경로)",
    })
    (model_dir / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
