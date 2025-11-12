<!-- LOGO -->
<p align="center">
  <img src="./team_logo.png" alt="bc-viz logo" width="220">
</p>

<h1 align="center">bc-viz</h1>

<p align="center">
  <em>Breast Cancer Wisconsin (Diagnostic) 데이터를 중심으로 분포·상관·패턴을 직관적으로 탐색하는 Python 기반 시각화 프로젝트</em>
</p>

---

bc-viz는 데이터의 분포, 상관관계, 패턴을 직관적으로 표현하는 Python 기반 시각화 프로젝트로, 분석 결과를 대시보드와 리포트 형태로 탐색할 수 있습니다.  
**bc-viz is a Python-based visualization project that intuitively displays data distributions, correlations, and patterns through interactive dashboards and reports.**

---

## 📦 Dataset at a Glance

> **Breast Cancer Wisconsin (Diagnostic)** (UCI Machine Learning Repository)  
> Source: Wisconsin Diagnostic Center  
> URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

| 항목 | 내용 |
|---|---|
| 데이터 이름 | Breast Cancer Wisconsin (Diagnostic) |
| 주제 분야 | 의학 — 유방암 진단 |
| 문제 유형 | 이진 분류 (양성 / 악성) |
| 표본 수 | 569 |
| 특징 수 | 30 (모두 연속형 실수형) |
| 타깃 라벨 | `diagnosis` (M = 악성, B = 양성) |
| 데이터 출처 | 위스콘신 대학 병리학 연구소 (Wisconsin Diagnostic Center) |

---

## 📚 Feature Dictionary

| 구분 | 변수명(name) | 의미(영문) | 설명(한글) | 측정 구분 |
|---:|---|---|---|---|
| 0 | ID | Identifier | 환자 또는 샘플 고유 번호 | 식별자 |
| 1 | Diagnosis | Diagnosis (M = malignant, B = benign) | 진단 결과 (M=악성, B=양성) | 타깃 변수 |
| 2 | radius1 | Mean Radius | 평균 반경 (세포 중심에서 경계까지의 평균 거리) | 평균(mean) |
| 3 | texture1 | Mean Texture | 평균 질감 (회색조 값의 표준편차) | 평균(mean) |
| 4 | perimeter1 | Mean Perimeter | 평균 둘레 길이 | 평균(mean) |
| 5 | area1 | Mean Area | 평균 면적 | 평균(mean) |
| 6 | smoothness1 | Mean Smoothness | 평균 매끄러움 (반경 길이의 지역적 변화) | 평균(mean) |
| 7 | compactness1 | Mean Compactness | 평균 조밀도 ((둘레² / 면적) - 1.0) | 평균(mean) |
| 8 | concavity1 | Mean Concavity | 평균 오목함의 정도 | 평균(mean) |
| 9 | concave_points1 | Mean Concave Points | 평균 오목한 부분의 개수 | 평균(mean) |
| 10 | symmetry1 | Mean Symmetry | 평균 대칭도 | 평균(mean) |
| 11 | fractal_dimension1 | Mean Fractal Dimension | 평균 프랙탈 차원 (“해안선 근사도”) | 평균(mean) |
| 12 | radius2 | Radius SE | 반경의 표준오차 | 표준오차(se) |
| 13 | texture2 | Texture SE | 질감의 표준오차 | 표준오차(se) |
| 14 | perimeter2 | Perimeter SE | 둘레의 표준오차 | 표준오차(se) |
| 15 | area2 | Area SE | 면적의 표준오차 | 표준오차(se) |
| 16 | smoothness2 | Smoothness SE | 매끄러움의 표준오차 | 표준오차(se) |
| 17 | compactness2 | Compactness SE | 조밀도의 표준오차 | 표준오차(se) |
| 18 | concavity2 | Concavity SE | 오목함의 표준오차 | 표준오차(se) |
| 19 | concave_points2 | Concave Points SE | 오목한 부분의 표준오차 | 표준오차(se) |
| 20 | symmetry2 | Symmetry SE | 대칭도의 표준오차 | 표준오차(se) |
| 21 | fractal_dimension2 | Fractal Dimension SE | 프랙탈 차원의 표준오차 | 표준오차(se) |
| 22 | radius3 | Worst Radius | 최댓값 반경 | 최댓값(worst) |
| 23 | texture3 | Worst Texture | 최댓값 질감 | 최댓값(worst) |
| 24 | perimeter3 | Worst Perimeter | 최댓값 둘레 | 최댓값(worst) |
| 25 | area3 | Worst Area | 최댓값 면적 | 최댓값(worst) |
| 26 | smoothness3 | Worst Smoothness | 최댓값 매끄러움 | 최댓값(worst) |
| 27 | compactness3 | Worst Compactness | 최댓값 조밀도 | 최댓값(worst) |
| 28 | concavity3 | Worst Concavity | 최댓값 오목함 | 최댓값(worst) |
| 29 | concave_points3 | Worst Concave Points | 최댓값 오목한 부분의 개수 | 최댓값(worst) |
| 30 | symmetry3 | Worst Symmetry | 최댓값 대칭도 | 최댓값(worst) |
| 31 | fractal_dimension3 | Worst Fractal Dimension | 최댓값 프랙탈 차원 | 최댓값(worst) |

---

## 🧭 Project Goals (요약)
- 분포, 상관, 이상치 탐색을 위한 **EDA 대시보드** 제공
- 모델링/리포팅에 바로 쓰는 **자동 리포트 아티팩트** 생성

## 📝 Citation
If you use this dataset, please cite the UCI ML Repository entry:  
*Breast Cancer Wisconsin (Diagnostic) Data Set, UCI Machine Learning Repository.*

