from __future__ import annotations

# ====================================================
# 52번. 신뢰도 보정: 모델 확률 → "이 예측이 맞을 확률"
# ====================================================
# Random Forest의 1위 확률(17개 동작에 나눠 준 몫)은 실제 정답률보다 훨씬 낮게 나옵니다
# (사람 교차 검증: 정답률 0.90인데 평균 1위 확률 0.42).
# 51번 --oof가 저장한 "처음 보는 사람" 예측으로, 1위 확률·2위와의 차이 등에서
# 실제 정답 여부를 맞히는 로지스틱 회귀(보정기)를 학습합니다.
#
# 평가도 사람 단위로 나눕니다: 보정기를 4명으로 학습 → 남은 1명으로 확인 (5회).
# 결과: reports/motion_capture_aug/calib/calibration.txt
#       artifacts/motion_capture_aug/calibration.json (서비스가 읽는 계수)
#
# 사용: .venv/bin/python scripts/52_fit_confidence_calibration.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "reports" / "motion_capture_aug" / "calib" / "oof_predictions.csv"
OUT_TXT = ROOT / "reports" / "motion_capture_aug" / "calib" / "calibration.txt"
OUT_JSON = ROOT / "artifacts" / "motion_capture_aug" / "calibration.json"
FEATURES = ["p1", "p2", "margin", "log_ratio", "entropy"]


def features(proba: np.ndarray) -> np.ndarray:
    """확률 벡터들(N×17) → 보정기 입력 (서비스 action_inference.calibrated_confidence와 같은 계산)."""
    srt = -np.sort(-proba, axis=1)
    p1, p2 = srt[:, 0], srt[:, 1]
    eps = 1e-6
    entropy = -(proba * np.log(proba + eps)).sum(1) / np.log(proba.shape[1])
    return np.c_[p1, p2, p1 - p2, np.log((p1 + eps) / (p2 + eps)), entropy]


def ece(prob, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            total += m.mean() * abs(prob[m].mean() - correct[m].mean())
    return total


def main():
    df = pd.read_csv(OOF)
    classes = [c for c in df.columns if c.startswith("A")]
    proba = df[classes].to_numpy()
    pred = np.array(classes)[proba.argmax(1)]
    correct = (pred == df["action"].to_numpy()).astype(int)
    x = features(proba)
    subjects = df["subject"].to_numpy()

    # 사람 단위 교차 평가
    calibrated = np.zeros(len(df))
    for s in np.unique(subjects):
        tr, te = subjects != s, subjects == s
        model = LogisticRegression(C=1.0, max_iter=1000).fit(x[tr], correct[tr])
        calibrated[te] = model.predict_proba(x[te])[:, 1]
    raw = x[:, 0]

    lines = [f"OOF 예측 {len(df)}건 (조건별 {df['condition'].value_counts().to_dict()}), 정답률 {correct.mean():.3f}", ""]
    lines.append("방법            평균표시값  실제정답률  ECE(낮을수록 정확)  Brier  AUC(맞음/틀림 구분)")
    for name, prob in (("모델 1위 확률", raw), ("보정 후", calibrated)):
        lines.append(f"{name:14s} {prob.mean():10.3f} {correct.mean():10.3f} {ece(prob, correct):14.3f} "
                     f"{brier_score_loss(correct, prob):8.3f} {roc_auc_score(correct, prob):8.3f}")
    lines.append("")
    lines.append("조건별 (보정 후 평균 / 실제 정답률)")
    for cond in sorted(df["condition"].unique()):
        m = df["condition"].to_numpy() == cond
        lines.append(f"  {cond:6s} {calibrated[m].mean():.3f} / {correct[m].mean():.3f}   (원래 1위 확률 평균 {raw[m].mean():.3f})")

    lines.append("")
    lines.append("구간별 신뢰도 (보정 후 값 구간 → 실제 정답률, 건수)")
    for lo, hi in ((0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)):
        m = (calibrated >= lo) & (calibrated < hi)
        if m.any():
            lines.append(f"  {lo:.2f}~{min(hi, 1):.2f}: 실제 {correct[m].mean():.3f} ({m.sum()}건)")

    lines.append("")
    lines.append("'불확실' 기준 후보 (보정 후 값 < 기준이면 불확실)")
    lines.append("  기준   불확실 비율   확실로 표시된 것의 정답률   틀린 예측 중 불확실로 걸러진 비율")
    for th in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        unc = calibrated < th
        sure_acc = correct[~unc].mean() if (~unc).any() else float("nan")
        caught = (unc & (correct == 0)).sum() / max(1, (correct == 0).sum())
        lines.append(f"  {th:.2f}   {unc.mean():8.2f}   {sure_acc:20.3f}   {caught:24.2f}")
    lines.append("")
    lines.append("참고: 기존 기준(모델 1위 확률 < 0.4)")
    unc = raw < 0.4
    lines.append(f"  불확실 비율 {unc.mean():.2f}, 확실 정답률 {correct[~unc].mean():.3f}, "
                 f"틀린 것 중 걸러진 비율 {(unc & (correct == 0)).sum() / max(1, (correct == 0).sum()):.2f}")

    # 서비스용: 전체 데이터로 다시 학습한 계수
    final = LogisticRegression(C=1.0, max_iter=1000).fit(x, correct)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "method": "logistic_regression",
        "features": FEATURES,
        "coef": [round(float(c), 6) for c in final.coef_[0]],
        "intercept": round(float(final.intercept_[0]), 6),
        "n_samples": int(len(df)),
        "base_accuracy": round(float(correct.mean()), 4),
        "source": "51번 --oof 사람 교차 검증 예측 (clean·wc15·wc60), 52번 학습",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines.append(f"\n저장: {OUT_JSON}")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
