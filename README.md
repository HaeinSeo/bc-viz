<h1 align="center" style="font-family: 'Comic Sans MS', 'Garamond', cursive; color:#993A6B;">
💗 bc-viz 💗
</h1>

<p align="center">
  <img src="./team_logo.png" width="180" alt="bc-viz Logo">
</p>

<p align="center" style="font-size:16px; line-height:1.6;">
🧬 <b>bc-viz</b>는 <b>Breast Cancer Wisconsin (Diagnostic)</b> 데이터를 기반으로  
유방암 진단과 관련된 <b>특징(feature) 분포, 상관관계, 패턴</b>을 시각적으로 탐색할 수 있도록 설계된  
<b>Python 기반 데이터 시각화 프로젝트</b>입니다.  
데이터의 구조적 관계를 직관적으로 보여주며,  
분석 결과를 <b>대시보드</b> 및 <b>리포트</b> 형태로 탐색할 수 있습니다. 🌷
</p>

---

### 🩺 About the Dataset

The project utilizes the  
<a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic">
<b>UCI “Breast Cancer Wisconsin (Diagnostic)” dataset</b></a>,  
collected by the <i>University of Wisconsin Diagnostic Center</i>.  
It contains 569 samples with 30 continuous numerical features  
used to classify tumors as **malignant (M)** or **benign (B)**.

---

### 💡 Tech Stack
🐍 `Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Dash`

---

### 📊 Dataset Overview

| 항목 | 내용 |
|:---|:---|
| **데이터 이름** | Breast Cancer Wisconsin (Diagnostic) |
| **주제 분야** | 의학 — 유방암 진단 |
| **문제 유형** | 이진 분류 (양성 / 악성) |
| **표본 수** | 569 |
| **특징 수** | 30 (모두 연속형 실수형) |
| **타깃 라벨** | `diagnosis` (M = 악성, B = 양성) |
| **데이터 출처** | 위스콘신 대학 병리학 연구소 (Wisconsin Diagnostic Center) |

---

### 🔎 Feature Dictionary

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
| 22 | radius3 | Worst Radius | 최악 상태 반경 | 최악 상태(worst) |
| 23 | texture3 | Worst Texture | 최악 상태 질감 | 최악 상태(worst) |
| 24 | perimeter3 | Worst Perimeter | 최악 상태 둘레 | 최악 상태(worst) |
| 25 | area3 | Worst Area | 최악 상태 면적 | 최악 상태(worst) |
| 26 | smoothness3 | Worst Smoothness | 최악 상태 매끄러움 | 최악 상태(worst) |
| 27 | compactness3 | Worst Compactness | 최악 상태 조밀도 | 최악 상태(worst) |
| 28 | concavity3 | Worst Concavity | 최악 상태 오목함 | 최악 상태(worst) |
| 29 | concave_points3 | Worst Concave Points | 최악 상태 오목한 부분 개수 | 최악 상태(worst) |
| 30 | symmetry3 | Worst Symmetry | 최악 상태 대칭도 | 최악 상태(worst) |
| 31 | fractal_dimension3 | Worst Fractal Dimension | 최악 상태 프랙탈 차원 | 최악 상태(worst) |

---

### 👥 Team Members

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./haein.png" width="120"><br/>
        <b>서해인</b>
      </td>
      <td align="center">
        <img src="./songhee.png" width="120"><br/>
        <b>김송희</b>
      </td>
      <td align="center">
        <img src="./soomin.png" width="120"><br/>
        <b>서수민</b>
      </td>
      <td align="center">
        <img src="./yuzi.png" width="120"><br/>
        <b>이유지</b>
      </td>
    </tr>
  </table>
</p>


---

### 🎯 Vision
<p align="center" style="font-size:15px; line-height:1.7;">
bc-viz는 <b>의료 데이터 시각화</b>를 통해  
AI 기반 진단 보조 시스템의 가능성을 제시하고,  
데이터 과학의 <b>해석 가능성(Explainability)</b>을 높이는 것을 목표로 합니다. 🩷  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Data%20Visualization-Matplotlib-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Medical%20Analytics-Breast%20Cancer-pink?style=for-the-badge">
</p>

---

### 📚 Citation
If you use this dataset, please cite the UCI ML Repository entry:  
**Breast Cancer Wisconsin (Diagnostic) Data Set, UCI Machine Learning Repository.**
