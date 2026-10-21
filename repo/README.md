# Machine-Knitted QR

편직 QR코드의 인식과 정사각형 격자 보존도 — 분석 코드와 측정 자료
Square-grid preservation and the recognition of machine-knitted QR codes — code and data

차유나(You-na Cha), 박진완(Jin-wan Park) · 중앙대학교
한국섬유공학회지 투고 원고의 분석 코드와 측정값입니다.

## 연구 요약

가정용 단일 베드 수편기(Brother KH-910)로 2색 페어아일 QR코드(버전 3, 오류정정 H)를 편성했습니다. 모듈 1칸은 2코 × 3단의 정수 매핑이며, 도안은 74코 × 111단입니다. 편직물 16장은 무장력 상태와 평면 6개 고정 상태에서 촬영했습니다. 사진은 설계 격자에 정합해 아래 세 값을 계산했습니다.

- 격자 잔차: 원근 보정 후 남는 격자점 변위의 RMS
- 파인더 영역 최대 변위
- 모듈 오류율

| 결과 | 값 |
|---|---|
| 모듈 오류율 (조건 평균) | 0.0–1.5% |
| 무장력 상태에서 인식된 편직물 | 0장 |
| 고정 후 파인더 최대 변위 감소 | 15/15장 (부호 검정 p < 0.001) |
| 고정 후 인식이 실패에서 성공으로 바뀜 | 16장 중 14장 |
| 편직물을 넘나드는 단일 임계값 | 재현되지 않음 |

## 폴더 구성

```
code/
  make_chart.py             URL → 정수 매핑 편직 도안(겉면/기계 입력용 좌우 반전)
  knitqr_analyze.py         사진 → 격자 잔차·파인더 변위·모듈 오류율 (CLI)
  synthetic_validation.py   크기를 아는 변위장을 넣은 합성 영상 35장으로 정합 정확도 검증
  decoder_reread.py         ZXing-C++ · ZBar · OpenCV · WeChat 재판독
  finder_sensitivity.py     파인더 지표: 최댓값 vs RMS vs 90백분위수
design/                     사용한 도안 PNG (as seen / mirrored for machine)
data/                       측정값 CSV (아래 표)
figures/                    논문 Figure 1–4
```

## 자료 (data/)

| 파일 | 내용 |
|---|---|
| photo_manifest.csv | 재판독 대상 사진 143장: 측정 단계(ds), 편직물(fab), 상태(st), 파일명, 스마트폰 앱 인식(phone) |
| geometry_per_photo_131.csv | 기하 분석에 쓴 사진 131장의 격자 잔차(rms)·파인더 최대 변위(fin)·인식(rec) |
| first_photography_region.csv | 1차 촬영 사진별 영역 지표 (C2 = 무장력 집게, C3 = 장력) |
| ruler_2nd_measurement_S1-S5.csv | 2차 측정 자 길이 (세로 좌/중/우, 가로 위/중/아래) |
| B_batch_ruler_and_indices.csv | B1–B5 자 길이와 사진 지표 |
| B5_hanging_load.csv | B5 수직 하중 단계별 세로 가운데 길이 |
| module_error_later_photos.csv | 2차 측정 이후 사진의 모듈 오류율·정합 신뢰도(NCC) |
| synthetic_validation.csv | 합성 영상 35장의 참값·추정값 |
| decoder_reread.csv | 사진별 공개 디코더 4종 해독 여부 |
| finder_sensitivity_matched.csv | 파인더 지표 민감도 분석 (A5 무장력 1장 제외 130장) |
| SupplementaryTable_S1.xlsx | 15장 고정 전후 중앙값과 감소량 |

상태 표기는 다음과 같습니다.
- clamp: 무장력 집게
- taut: 장력
- free: 무장력
- six: 6개 고정
- seven: 7개 고정
- side: 좌우 고정

측정 단계(ds)는 1st(1차 촬영), 2nd(2차 측정), ver(수리 후 검증 묶음과 B묶음)로 나뉩니다.

## 실행

```
pip install -r requirements.txt
python code/make_chart.py --url https://zrr.kr/ZFyyPM --version 3 --ecc H --sx 2 --sy 3 --border 4
python code/knitqr_analyze.py 사진폴더 --chart design/QR_knit_chart_74x111_as_seen.png --sx 2 --sy 3 --border 4
python code/synthetic_validation.py
```

- `make_chart.py`의 출력은 `design/`의 도안과 화소 단위로 같습니다.
- `knitqr_analyze.py`는 사진 1장에 약 5초가 걸립니다. 좌우 반전으로 편성된 편직물은 자동으로 감지합니다.
- `decoder_reread.py`와 `finder_sensitivity.py`는 `photos/` 폴더에 원본 사진이 있어야 합니다. WeChat 검출기는 opencv_3rdparty의 wechat_qrcode 모델 4개를 `code/wechat_models/`에 두면 됩니다.

### 주요 파라미터 (knitqr_analyze.py)

| 단계 | 값 |
|---|---|
| 사진 크기 | 긴 변 1600 px로 축소 |
| 정규화 좌표 | 모듈당 16 px |
| 격자점 | 3모듈 간격 |
| 블록 매칭 | 6 × 6 모듈 템플릿, ±1.5 모듈 탐색, 정규화 상호상관 |
| 호모그래피 | RANSAC 허용 오차 0.75 모듈, 2회 재추정 |
| 국소 재배치 | 역거리 가중 보간 |
| 측정 제외 기준 | 정합 신뢰도(NCC) 0.5 미만 |

## 원본 사진

촬영 원본 사진은 개인 공간과 앱 화면이 함께 찍혀 있어 이 저장소에 넣지 않았습니다. 연구 목적의 요청은 교신저자(jinpark@cau.ac.kr)에게 문의해 주세요.

## Citation

논문 게재 후 서지 정보를 추가합니다.
