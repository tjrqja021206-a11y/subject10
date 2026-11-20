# Motion Capture 데이터

## 파일 이름 규칙

`S01_A001_T01.mp4` = **S**ubject(사람) · **A**ction(동작) · **T**ake(촬영)

- `S01` : 촬영 대상자 번호
- `A001` ~ `A017` : 동작 번호 (아래 표)
- `T01` / `T02` : 같은 동작을 **다른 방향**에서 찍은 영상

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

## S01 현황

- 영상 32개, 60fps
- A016, A017은 `T01`만 있음
- A009, A010, A012의 `T02`는 가로 영상, 나머지는 세로 영상

## S02 현황

- 영상 34개, 60fps, A001~A017 모두 `T01`/`T02` 있음
- A009, A010, A012, A016, A017의 `T02`는 가로 영상, 나머지는 세로 영상
- 이름 없던 영상(`-.mp4`, 18초)은 `_unsorted/S02_unknown.mp4`로 따로 보관

## 폴더 구성

```
data/
├── motion_capture/            # 원본 영상 (git 제외)
│   ├── S01/S01_A001_T01.mp4 ...
│   ├── S02/S02_A001_T01.mp4 ...
│   └── _unsorted/             # 동작을 모르는 영상
├── motion_capture_keypoints/  # 스켈레톤 결과
│   ├── manifest_S01.csv       # 영상별 요약 (프레임 수, 검출률, 신뢰도 등)
│   ├── S01/S01_A001_T01.csv ...
│   └── S02/S02_A001_T01.csv ...
└── motion_capture_overlay/    # 스켈레톤 확인 영상 (git 제외)
```

새 사람의 영상은 `data/motion_capture/S03/`처럼 원본 폴더에 넣습니다
(`motion_capture_overlay`는 47번 결과 폴더입니다).

스켈레톤 CSV는 영상과 같은 이름이며, 한 행이 한 프레임입니다
(COCO 17관절 × 픽셀 좌표·정규화 좌표·신뢰도, 총 95컬럼).

## 스크립트

```bash
# 1) 영상 → 스켈레톤 CSV (data/motion_capture_keypoints/S01/)
.venv/bin/python scripts/46_extract_motion_capture_keypoints.py --subject S01

# 2) 스켈레톤을 영상 위에 그려서 확인 (data/motion_capture_overlay/S01/, git 제외)
.venv/bin/python scripts/47_render_motion_capture_skeleton.py --subject S01

# 3) 17개 동작 구분 테스트 (reports/motion_capture/)
.venv/bin/python scripts/48_test_motion_capture_actions.py
```

## 동작 구분 테스트 결과 (S01 + S02)

영상 단위 다수결 정확도, 우연히 맞을 확률 약 6%

| 테스트 | seq (기존 좌표+각도) | robust (각도·높이, 좌우 무관) |
|---|---|---|
| 방향 교차 T01 → T02 | 19% | 13% |
| 방향 교차 T02 → T01 | 9% | 38% |
| 사람 교차 S02 → S01 | 53% | 59% |
| 사람 교차 S01 → S02 | 53% | 71% |

- 한 방향만 학습하면 다른 방향은 거의 못 맞힘 → 학습에 양방향을 모두 넣어야 함
- 자주 틀리는 동작: 쪼그려 물건 빼내기, 샷건, 키오스크 두드리기
- 사람이 늘어나면(S03…) 48번이 자동으로 한 명씩 빼고 테스트함
