# =============================================
# 🧬 Breast Cancer Wisconsin (Diagnostic) 데이터셋 전체 pandas DataFrame으로 불러오기
# =============================================

from ucimlrepo import fetch_ucirepo
import pandas as pd

# 1. 데이터셋 로드 (UCI ID = 17)
dataset = fetch_ucirepo(id=17)

# 2️. Feature(X), Target(y) 결합
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# 3️. DataFrame 확인
print("전체 데이터셋 형태:", df.shape)
print("\n상위 5개 행 미리보기:")
print(df.head())

# 4️. 결측치, 타입, 요약 정보
print("\nDataFrame 정보:")
print(df.info())

# 5️. CSV로 저장 (선택)
df.to_csv("breast_cancer_wisconsin_diagnostic.csv", index=False)
print("\nCSV 파일 저장 완료: breast_cancer_wisconsin_diagnostic.csv")
