# 심리 성향 예측 AI 해커톤

> 데이콘 | 이진 분류 | 평가 지표: AUC | **최종 점수: 0.78182**

## 대회 개요

설문 기반 심리·행동 데이터를 활용해 개인의 **투표 여부(voted)** 를 예측하는 AI 모델 개발.

- 마키아벨리즘 성향 문항 응답 (QaA ~ QtA, 20문항)
- 문항별 응답 소요 시간 (QaE ~ QtE)
- Big Five 성격 지표 (tp01 ~ tp10)
- 인구통계·환경 정보 (나이, 성별, 종교, 인종 등)

## 접근 방법

### 피처 엔지니어링

- **역코딩**: 마키아벨리즘 역방향 문항(QeA, QfA, QkA, QqA, QrA) 및 통계적으로 음의 상관관계를 보이는 비공개 문항에 역코딩 적용
- **마키아벨리즘 하위척도**: 기만성(T), 의지력(V), 냉담함(M) 점수 파생
- **Big Five 점수**: 외향성, 친화성, 성실성, 정서안정성, 개방성 계산
- **응답 지연 통계**: 합산, 최대값, 표준편차, 왜도
- **교호작용 피처**: 마키아벨리즘 × 연령대, 마키아벨리즘 × 정서안정성

### 모델 구성

| 모델 | 알고리즘 | 주요 기법 |
|------|----------|-----------|
| Model 1 | VotingClassifier | RF + LGBM + GBM Soft Voting |
| Model 2 | LightGBM | Optuna 하이퍼파라미터 튜닝 + RFE × 4 시드 앙상블 |
| Model 3 | Neural Network | ResidualBlock + Label Smoothing + SWA (Stochastic Weight Averaging) |
| **최종** | **Weighted Ensemble** | **Optuna로 최적 가중치 탐색** |

## 결과

| 모델 | OOF AUC |
|------|---------|
| Model 1 (Voting) | - |
| Model 2 (LGBM RFE) | - |
| Model 3 (NN) | - |
| **최종 앙상블** | **0.78182** |

## 프로젝트 구조

```
psychological-voting-prediction/
├── data/                   # 원본 데이터 (데이콘에서 다운로드)
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── final_solution.ipynb  # EDA + 전체 모델 파이프라인
├── src/
│   ├── preprocess.py       # 피처 엔지니어링 함수
│   ├── train.py            # 모델 학습 (Model 1·2·3)
│   └── predict.py          # 앙상블 가중치 탐색 + 제출 파일 생성
├── docs/
│   └── presentation.pptx   # 발표 자료
├── models/                 # 학습된 모델 파일 (gitignore)
├── outputs/figures/        # EDA 시각화 결과물
├── submission/             # 최종 제출 파일
├── requirements.txt
└── .gitignore
```

## 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. data/ 폴더에 데이콘 데이터 배치
#    https://dacon.io 에서 해당 대회 데이터 다운로드

# 3. 전체 파이프라인 실행 (학습 + 예측)
cd src
python predict.py
# → submission/submission.csv 생성
```

## 팀 구성

1조 — 이런 경우 저런 서영
