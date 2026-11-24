# Motion Capture 데이터

## 파일 이름 규칙

`S01_A001_T01.mp4` = **S**ubject(사람) · **A**ction(동작) · **T**ake(촬영 방향)

- `S01` ~ `S05` : 촬영 대상자 번호
- `A001` ~ `A017` : 동작 번호 (아래 표)
- 촬영 방향
  - `T01` : 앞
  - `T02` : 오른쪽 옆
  - `T03` : 뒤
  - `T02M` : T02를 좌우 반전해 만든 "왼쪽" 증강 데이터 (49번, 스켈레톤 CSV만 있음, 학습 전용)

## 동작 목록

| 번호 | 동작 | action_label |
|---|---|---|
| A001 | 키오스크 두드리기 | kiosk_tap |
| A002 | 한손 물건 던지기 | throw_one_hand |
| A003 | 앞차기 | front_kick |
| A004 | 잡고 흔들기 | grab_shake |
| A005 | 밀치기 | push |
| A006 | 휘두르기 | swing |
| A007 | 바닥에 앉기 | sit_floor |
| A008 | 위에 앉기 | sit_on_object |
| A009 | 앞으로 눕기 | lie_forward |
| A010 | 뒤로 눕기 | lie_backward |
| A011 | 몸통 박치기 | body_slam |
| A012 | 쪼그려 물건 빼내기 | crouch_take_item |
| A013 | 샷건 | shotgun |
| A014 | 구토 | vomit |
| A015 | 옆으로 쓰러지기 | fall_sideways |
| A016 | 뒤로 넘어지기 | fall_backward |
| A017 | 앞으로 넘어지기 | fall_forward |

`action_label`은 [configs/motion_capture_action_classes.csv](../../configs/motion_capture_action_classes.csv)와 같습니다.

## 데이터 현황

- S01 ~ S05, 사람마다 17개 동작 × 3방향 = 51개, 모두 255개 (60fps)
- 증강: T02 좌우 반전 85개 (`data/motion_capture_keypoints_aug/`)
- 방향별 평균 관절 신뢰도: 앞 0.93, 오른쪽 0.83, 뒤 0.75 (뒤에서는 얼굴이 안 보임)
- 사람 검출이 일부 빠지거나 여러 명이 잡힌 영상: S02_A010_T01(검출 89%), S03_A009_T02, S04_A017_T02(화면 가장자리에 다른 사람) — 스켈레톤은 실제 동작자에게 그려짐

## 폴더 구성

```
data/
├── motion_capture/                # 원본 영상 (git 제외)
│   └── S01/S01_A001_T01.mp4 ... (S01 ~ S05)
├── motion_capture_keypoints/      # 스켈레톤 결과 (46번)
│   ├── manifest_S01.csv ...       # 영상별 요약 (프레임 수, 검출률, 신뢰도 등)
│   └── S01/S01_A001_T01.csv ...
├── motion_capture_keypoints_aug/  # 좌우 반전 증강 (49번)
│   └── S01/S01_A001_T02M.csv ...
└── motion_capture_overlay/        # 스켈레톤 확인 영상 (47번, git 제외)
artifacts/motion_capture/          # 최종 학습 모델 (48번)
reports/motion_capture/            # 테스트 결과 (48번)
```

새 사람의 영상은 `data/motion_capture/S06/`처럼 원본 폴더에 넣습니다.

스켈레톤 CSV는 영상과 같은 이름이며, 한 행이 한 프레임입니다
(COCO 17관절 × 픽셀 좌표·정규화 좌표·신뢰도, 총 95컬럼).

## 스크립트

```bash
# 1) 영상 → 스켈레톤 CSV
.venv/bin/python scripts/46_extract_motion_capture_keypoints.py --subject S01

# 2) 스켈레톤을 영상 위에 그려서 확인 (git 제외)
.venv/bin/python scripts/47_render_motion_capture_skeleton.py --subject S01

# 3) T02 좌우 반전 증강 (모든 Subject)
.venv/bin/python scripts/49_mirror_motion_capture_keypoints.py

# 4) 학습·테스트 + 최종 모델 저장
.venv/bin/python scripts/48_test_motion_capture_actions.py
```

## 학습 결과 (S01 ~ S05, 2026-11-23)

영상 단위 다수결 정확도, 우연히 맞을 확률 약 6%

### 사람 교차 (한 명씩 빼고 학습 → 뺀 사람으로 테스트)

| 조건 | 전체 | 앞 | 오른쪽 | 뒤 |
|---|---|---|---|---|
| seq (좌표+각도) | 74% | 76% | 72% | 74% |
| seq + 증강 | 74% | 74% | 73% | 74% |
| robust (각도·높이, 좌우 무관) | **81%** | 91% | 84% | 69% |
| robust + 증강 | **81%** | 92% | 80% | 72% |
| robust2 (+ 팔다리 뻗은 거리·손발 속도) | 84% | 89% | 86% | 76% |
| robust2 + 증강 | 83% | 89% | 84% | 76% |
| hybrid (robust2 창 모델 + 영상 전체 모델, 최종 모델) | **91%** | 92% | 92% | 88% |
| hybrid + 증강 | 90% | 92% | 93% | 85% |

사람별(hybrid): S01 82%, S02 94%, S03 92%, S04 94%, S05 90%

hybrid는 2초 창 확률 평균에 "영상 전체 요약"(10구간 자세 흐름 + 엉덩이·코·어깨가
가장 빨리 내려간/올라간 속도와 시점)을 보는 Clip 모델 확률을 더해 판정합니다.
→ 영상의 시작·끝이 정해진 경우(영상 단위 판정)에만 쓸 수 있고, 실시간 Sliding Window에는
  창 모델(`action_model.joblib`)만 쓸 수 있습니다. 영상 모델은 `clip_model.joblib`.

### 방향 교차 (한 방향을 빼고 학습 → 뺀 방향으로 테스트)

| 조건 | 앞 | 오른쪽 | 뒤 |
|---|---|---|---|
| seq | 20% | 14% | 18% |
| robust | 51% | 31% | 49% |
| robust2 | 51% | 34% | 53% |
| hybrid | 68% | 45% | 52% |

- 학습에 없는 방향은 여전히 약함 → 실제로 쓸 카메라 방향의 영상이 학습에 있어야 함
- 좌우 반전 증강은 효과가 거의 없음: robust 특징은 애초에 좌우를 구분하지 않아서 반전 데이터가 원본과 같고,
  seq에서도 차이가 없었음

### 동작별 (hybrid, 사람 교차, 15개 영상 중 맞힌 수)

- 15/15: 키오스크 두드리기, 앞차기, 휘두르기, 바닥에 앉기, 위에 앉기, 뒤로 눕기, 몸통 박치기, 구토
- 13~14: 한손 물건 던지기, 앞으로 눕기, 옆으로 쓰러지기, 뒤로 넘어지기, 앞으로 넘어지기
- 10~12: 잡고 흔들기(→ 몸통 박치기), 쪼그려 물건 빼내기(→ 앞으로 눕기), 샷건(→ 휘두르기)
- 9: 밀치기(→ 잡고 흔들기)

개선 과정 (15개 중 맞힌 수, robust+증강 → robust2 → hybrid)
- 팔다리 동작: 샷건 7→11→11, 앞차기 9→13→15, 던지기 8→11→14
- 바닥 동작: 앞으로 넘어지기 11→9→14, 쪼그려 빼내기 10→8→10, 옆으로 쓰러지기 12→13→14
- 남은 과제: 밀치기(9)와 잡고 흔들기·몸통 박치기 구분, 쪼그려 빼내기(10)
