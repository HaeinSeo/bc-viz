"""
BC-Viz: 유방암 데이터 분석 웹 대시보드
Flask 기반 HTML/CSS/JS 웹 애플리케이션
사진의 대시보드 디자인을 기반으로 한 모던한 인터페이스
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

app = Flask(__name__, static_folder='static', template_folder='templates')

# 전역 변수
df = None
X_scaled = None
y = None
feature_cols = None
model_cache = {}

# 데이터 로드 함수
def load_data():
    """데이터 로드 및 전처리"""
    global df, X_scaled, y, feature_cols
    
    if df is not None:
        return df, X_scaled, y, feature_cols
    
    try:
        if os.path.exists("kr_data.csv"):
            df = pd.read_csv("kr_data.csv", encoding="utf-8")
        elif os.path.exists("data.csv"):
            df = pd.read_csv("data.csv", encoding="utf-8")
        else:
            raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")
        
        # 타겟 컬럼 확인
        if "진단" in df.columns:
            target_col = "진단"
        elif "diagnosis" in df.columns:
            target_col = "diagnosis"
            df = df.rename(columns={"diagnosis": "진단"})
        else:
            raise ValueError("'진단' 또는 'diagnosis' 컬럼을 찾을 수 없습니다.")
        
        # 진단 인코딩
        if df[target_col].dtype == "object":
            df[target_col] = df[target_col].map({
                "M": 1, "B": 0,
                "악성(M)": 1, "양성(B)": 0,
                "악성": 1, "양성": 0
            })
        
        # Feature 컬럼 선택
        feature_cols = [c for c in df.columns 
                       if c not in ["id", "ID", target_col, "Unnamed: 32"]]
        feature_cols = [c for c in feature_cols 
                       if df[c].dtype in [np.int64, np.float64]]
        
        # 데이터 전처리
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # NaN 처리
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        return df, X_scaled, y, feature_cols
    
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None, None, None, None

# 로고를 base64로 인코딩
def encode_logo():
    """로고를 base64로 인코딩"""
    try:
        if os.path.exists("team_logo.png"):
            with open("team_logo.png", "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        pass
    return None

# 팀원 이미지를 base64로 인코딩
def encode_team_images():
    """팀원 캐릭터 이미지들을 base64로 인코딩"""
    team_images = {}
    team_members = ['yuzi', 'soomin', 'songhee', 'haein']
    
    for member in team_members:
        img_path = os.path.join("team_character", f"{member}.png")
        try:
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_file:
                    team_images[member] = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"팀원 이미지 로드 오류 ({member}): {e}")
    
    return team_images

@app.route('/')
def index():
    """메인 페이지"""
    # 데이터 로드
    load_data()
    logo_base64 = encode_logo()
    team_images = encode_team_images()
    
    # 디버깅: 이미지 로드 확인
    print(f"로드된 팀원 이미지 수: {len(team_images)}")
    for member, img_data in team_images.items():
        print(f"  - {member}: {len(img_data) if img_data else 0} bytes")
    
    return render_template('dashboard.html', logo_base64=logo_base64, team_images=team_images)

@app.route('/api/data/overview')
def get_overview():
    """데이터 개요 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    benign_count = int((y == 0).sum())
    malignant_count = int((y == 1).sum())
    malignant_pct = round((y == 1).sum() / len(y) * 100, 2)
    
    return jsonify({
        "total_samples": len(df),
        "benign_count": benign_count,
        "malignant_count": malignant_count,
        "malignant_pct": malignant_pct,
        "feature_count": len(feature_cols)
    })

@app.route('/api/data/diagnosis-distribution')
def get_diagnosis_distribution():
    """진단 분포 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    benign_count = int((y == 0).sum())
    malignant_count = int((y == 1).sum())
    
    return jsonify({
        "labels": ["양성 (B)", "악성 (M)"],
        "values": [benign_count, malignant_count],
        "colors": ["#48BBB4", "#FF6B9D"]
    })

@app.route('/api/data/preview')
def get_data_preview():
    """데이터 미리보기 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    preview = df.head(10).fillna(0)
    
    return jsonify({
        "columns": preview.columns.tolist(),
        "data": preview.values.tolist()
    })

@app.route('/api/visualization/boxplot')
def get_boxplot():
    """Boxplot 데이터 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    features = request.args.getlist('features')
    if not features:
        features = feature_cols[:5]
    
    data = []
    for feature in features:
        if feature in X_scaled.columns:
            values = X_scaled[feature].tolist()
            data.append({
                "feature": feature,
                "values": values,
                "min": float(np.min(values)),
                "q1": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q3": float(np.percentile(values, 75)),
                "max": float(np.max(values))
            })
    
    return jsonify({"data": data})

@app.route('/api/visualization/histogram')
def get_histogram():
    """히스토그램 데이터 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    features = request.args.getlist('features')
    if not features:
        features = feature_cols[:3]
    
    data = []
    for feature in features:
        if feature in X_scaled.columns:
            benign_data = X_scaled[y == 0][feature].tolist()
            malignant_data = X_scaled[y == 1][feature].tolist()
            
            data.append({
                "feature": feature,
                "benign": benign_data,
                "malignant": malignant_data
            })
    
    return jsonify({"data": data})

@app.route('/api/ml/train')
def train_models():
    """머신러닝 모델 학습 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    # 모델 캐시 확인
    if 'models' in model_cache:
        return jsonify(model_cache['models'])
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
        )
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_accuracy = float(accuracy_score(y_test, rf_pred))
        rf_cm = confusion_matrix(y_test, rf_pred).tolist()
        
        # Decision Tree
        dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt_clf.fit(X_train, y_train)
        dt_pred = dt_clf.predict(X_test)
        dt_accuracy = float(accuracy_score(y_test, dt_pred))
        dt_cm = confusion_matrix(y_test, dt_pred).tolist()
        
        # Feature Importance
        rf_importance = rf_clf.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=False)
        
        results = {
            "rf_accuracy": rf_accuracy,
            "dt_accuracy": dt_accuracy,
            "rf_cm": rf_cm,
            "dt_cm": dt_cm,
            "feature_importance": {
                "features": importance_df.head(15)['Feature'].tolist(),
                "importance": importance_df.head(15)['Importance'].tolist()
            }
        }
        
        # 모델 캐시 저장
        model_cache['models'] = results
        model_cache['rf_model'] = rf_clf
        model_cache['dt_model'] = dt_clf
        model_cache['X_train'] = X_train
        model_cache['X_test'] = X_test
        model_cache['y_train'] = y_train
        model_cache['y_test'] = y_test
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/malignant/analyze')
def analyze_malignant():
    """악성 심각도 분석 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    try:
        # 악성 데이터 필터링
        if '진단' in df.columns:
            target_col = '진단'
        elif 'diagnosis' in df.columns:
            target_col = 'diagnosis'
        else:
            return jsonify({"error": "진단 컬럼을 찾을 수 없습니다."}), 500
        
        df_malignant = df.copy()
        if df_malignant[target_col].dtype == 'object':
            df_malignant['진단_encoded'] = df_malignant[target_col].map({
                'M': 1, 'B': 0,
                '악성(M)': 1, '양성(B)': 0,
                '악성': 1, '양성': 0
            })
        else:
            df_malignant['진단_encoded'] = df_malignant[target_col]
        
        df_malignant = df_malignant[df_malignant['진단_encoded'] == 1].copy()
        
        if len(df_malignant) == 0:
            return jsonify({"error": "악성 데이터를 찾을 수 없습니다."}), 500
        
        # 악성 데이터만으로 스케일링
        X_malig = df_malignant[feature_cols].values
        scaler_malig = StandardScaler()
        X_malig_scaled = scaler_malig.fit_transform(X_malig)
        
        # KMeans 클러스터링
        kmeans_malig = KMeans(n_clusters=2, random_state=42, n_init=10)
        malig_clusters = kmeans_malig.fit_predict(X_malig_scaled)
        
        # 클러스터 평균 계산
        cluster_means = []
        for cluster_id in range(2):
            cluster_data = X_malig_scaled[malig_clusters == cluster_id]
            cluster_mean = cluster_data.mean(axis=0)
            cluster_means.append(cluster_mean)
        
        # 고악성 클러스터 결정
        cluster_sums = [cm.sum() for cm in cluster_means]
        high_malignant_cluster = int(np.argmax(cluster_sums))
        
        # 타겟 변수 생성
        y_severity = (malig_clusters == high_malignant_cluster).astype(int)
        
        # 분포 계산
        low_severity_count = int((y_severity == 0).sum())
        high_severity_count = int((y_severity == 1).sum())
        
        # 모델 학습
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
            X_malig_scaled, y_severity, test_size=0.2, random_state=42, stratify=y_severity
        )
        
        rf_sev = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        rf_sev.fit(X_train_sev, y_train_sev)
        y_pred_sev = rf_sev.predict(X_test_sev)
        accuracy_sev = float(accuracy_score(y_test_sev, y_pred_sev))
        cm_sev = confusion_matrix(y_test_sev, y_pred_sev).tolist()
        
        # Feature Importance
        rf_importance = rf_sev.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_importance
        }).sort_values('Importance', ascending=False)
        
        return jsonify({
            "total_malignant": len(df_malignant),
            "low_severity_count": low_severity_count,
            "high_severity_count": high_severity_count,
            "high_severity_pct": round(high_severity_count / len(y_severity) * 100, 2),
            "model_accuracy": accuracy_sev,
            "confusion_matrix": cm_sev,
            "feature_importance": {
                "features": importance_df.head(15)['Feature'].tolist(),
                "importance": importance_df.head(15)['Importance'].tolist()
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dimension-reduction/pca')
def get_pca():
    """PCA 차원 축소 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    try:
        n_components = int(request.args.get('n_components', 2))
        n_components = min(max(n_components, 2), 3)
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_var = pca.explained_variance_ratio_.tolist()
        
        result = {
            "explained_variance": explained_var,
            "total_explained": float(sum(explained_var)),
            "data": X_pca.tolist(),
            "labels": y.tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dimension-reduction/tsne')
def get_tsne():
    """t-SNE 차원 축소 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    try:
        n_components = int(request.args.get('n_components', 2))
        perplexity = int(request.args.get('perplexity', 30))
        n_components = min(max(n_components, 2), 3)
        
        # 샘플링 (t-SNE는 계산이 오래 걸리므로)
        sample_size = min(500, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        tsne = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=perplexity
        )
        X_tsne = tsne.fit_transform(X_sample)
        
        return jsonify({
            "data": X_tsne.tolist(),
            "labels": y_sample.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/correlation/matrix')
def get_correlation_matrix():
    """상관관계 행렬 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    try:
        # 상관관계 행렬 계산
        corr_matrix = X_scaled.corr()
        
        # 처음 20개 특징만 사용 (너무 크면)
        selected_features = feature_cols[:20] if len(feature_cols) > 20 else feature_cols
        corr_subset = corr_matrix.loc[selected_features, selected_features]
        
        return jsonify({
            "features": selected_features,
            "matrix": corr_subset.values.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/features/list')
def get_features_list():
    """특징 목록 API"""
    df, X_scaled, y, feature_cols = load_data()
    
    if df is None:
        return jsonify({"error": "데이터를 로드할 수 없습니다."}), 500
    
    return jsonify({"features": feature_cols})

@app.route('/api/xai/shap')
def get_xai_shap():
    """XAI SHAP 분석 API"""
    if 'rf_model' not in model_cache:
        return jsonify({"error": "모델을 먼저 학습시켜주세요."}), 400
    
    try:
        rf_model = model_cache['rf_model']
        X_test = model_cache['X_test']
        feature_cols = load_data()[3]
        
        # SHAP 대신 Feature Importance 사용 (SHAP 라이브러리 설치 없이)
        # SHAP 값은 feature importance와 유사하게 계산
        importance = rf_model.feature_importances_
        
        # 테스트 샘플 하나에 대한 예측 기여도 근사치
        sample_idx = 0
        sample = X_test.iloc[sample_idx:sample_idx+1]
        prediction = rf_model.predict(sample)[0]
        
        # Feature importance를 SHAP 값으로 근사
        shap_values = importance / np.sum(importance) * 10  # 정규화 및 스케일링
        
        # 상위 15개만
        top_indices = np.argsort(np.abs(shap_values))[-15:][::-1]
        
        return jsonify({
            "features": [feature_cols[i] for i in top_indices],
            "importance": [float(shap_values[i]) for i in top_indices],
            "prediction": int(prediction),
            "sample_index": int(sample_idx)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/xai/lime')
def get_xai_lime():
    """XAI LIME 분석 API"""
    if 'rf_model' not in model_cache:
        return jsonify({"error": "모델을 먼저 학습시켜주세요."}), 400
    
    try:
        rf_model = model_cache['rf_model']
        X_test = model_cache['X_test']
        y_test = model_cache['y_test']
        feature_cols = load_data()[3]
        
        # LIME 대신 permutation importance 사용 (LIME 라이브러리 설치 없이)
        sample_idx = 0
        sample = X_test.iloc[sample_idx:sample_idx+1]
        true_label = int(y_test.iloc[sample_idx])
        prediction = int(rf_model.predict(sample)[0])
        
        # Permutation importance를 LIME 설명으로 근사
        # 각 특징을 제거하거나 변경했을 때의 영향 계산
        base_proba = rf_model.predict_proba(sample)[0][prediction]
        
        explanations = []
        for i, feature in enumerate(feature_cols[:15]):  # 상위 15개만
            sample_perm = sample.copy()
            sample_perm[feature] = X_test[feature].mean()  # 평균값으로 대체
            perm_proba = rf_model.predict_proba(sample_perm)[0][prediction]
            importance = float(base_proba - perm_proba)
            explanations.append({
                "feature": feature,
                "value": importance
            })
        
        # 중요도 순으로 정렬
        explanations.sort(key=lambda x: abs(x['value']), reverse=True)
        
        return jsonify({
            "prediction": "악성 (M)" if prediction == 1 else "양성 (B)",
            "true_label": "악성 (M)" if true_label == 1 else "양성 (B)",
            "explanation": explanations[:10]  # 상위 10개만
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 템플릿 폴더 확인 및 생성
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("BC-Viz 웹 대시보드 서버 시작...")
    print("브라우저에서 http://localhost:5000 접속하세요.")
    app.run(debug=True, host='0.0.0.0', port=5000)
