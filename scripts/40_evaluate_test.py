from __future__ import annotations

# ==================================================
# 40번. 최종 Test 성능 평가 (6일차 교재 기준)
# ==================================================
# 35번에서 S01로 학습한 Baseline Random Forest에
# 처음 보는 S03의 6개 Action Clip을 넣어 최종 성능을 확인합니다.
# model.fit()을 호출하지 않으므로 재학습하지 않습니다.

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==================================================
# ① 입력 경로: 34번 Test Sequence, 35번 모델·Feature 목록
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "test.csv"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_LIST_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)


# ==================================================
# ② 출력 경로: 지표 TXT, Clip별 예측 CSV, 혼동행렬 PNG
# ==================================================

METRIC_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_metrics.txt"
)

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_predictions.csv"
)

CONFUSION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "confusion"
    / "test_confusion_matrix.png"
)


# ==================================================
# ③ 학습 당시 Feature 이름과 순서 읽기
# ==================================================
# 30 Step × 38 Feature = 1,140개 이름을 같은 순서로 사용해야 합니다.

def load_feature_columns():
    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    # ==============================================
    # ④ 모델, Test CSV, Feature 목록 읽기
    # ==============================================
    # joblib 파일은 신뢰할 수 있는 출처의 것만 불러옵니다.

    model = joblib.load(
        MODEL_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    columns = (
        load_feature_columns()
    )

    # ==============================================
    # ⑤ 학습 Feature가 Test에 모두 있는지 확인
    # ==============================================

    missing = [
        column
        for column in columns
        if column not in test.columns
    ]

    if missing:
        raise RuntimeError(
            "Test에 학습 Feature가 "
            "누락되었습니다."
        )

    # ==============================================
    # ⑥ X_test(입력)와 y_true(정답) 분리
    # ==============================================
    # X_test.shape = (6, 1140)
    # action_label은 X_test에 넣지 않고 비교용 정답으로만 씁니다.

    X_test = (
        test[columns]
        .to_numpy(
            dtype=float
        )
    )

    y_true = (
        test[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    # ==============================================
    # ⑦ 예측과 Clip별 최고 예측확률
    # ==============================================
    # Confidence는 모델의 예측확률이며
    # 실제로 정답일 확률로 보정된 값은 아닙니다.

    y_pred = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    labels = list(
        model.classes_
    )

    # ==============================================
    # ⑧ Accuracy, Macro F1, Classification Report
    # ==============================================
    # Test가 6개뿐이라 Clip 하나가 점수를 크게 바꿉니다.

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )
    print(
        f"Test Macro F1: "
        f"{macro_f1:.4f}"
    )
    print()
    print(report)

    # ==============================================
    # ⑨ 지표 TXT 저장
    # ==============================================

    METRIC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_PATH.write_text(
        (
            f"Test Accuracy: "
            f"{accuracy:.6f}\n"
            f"Test Macro F1: "
            f"{macro_f1:.6f}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    # ==============================================
    # ⑩ Clip별 실제 정답·예측·정오답 CSV 저장
    # ==============================================

    prediction_df = (
        pd.DataFrame(
            {
                "clip_id": (
                    test["clip_id"]
                ),
                "subject": (
                    test["subject"]
                ),
                "true_label": (
                    y_true
                ),
                "pred_label": (
                    y_pred
                ),
                "confidence": (
                    confidence
                ),
                "correct": (
                    y_true == y_pred
                ),
            }
        )
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ==============================================
    # ⑪ 혼동행렬 PNG 저장 (행: 실제, 열: 예측)
    # ==============================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )
    )

    display.plot(
        xticks_rotation=45,
    )

    plt.title(
        "Test Confusion Matrix"
    )
    plt.tight_layout()

    CONFUSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_PATH,
        dpi=150,
    )
    plt.close()

    print(
        f"Metrics: "
        f"{METRIC_PATH}"
    )
    print(
        f"Predictions: "
        f"{PREDICTION_PATH}"
    )
    print(
        f"Confusion Matrix: "
        f"{CONFUSION_PATH}"
    )


if __name__ == "__main__":
    main()
