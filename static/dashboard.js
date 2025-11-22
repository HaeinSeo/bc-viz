// BC-Viz 대시보드 JavaScript
// 페이지 상태 관리
let currentPage = 'overview';
let darkMode = false;
let showMalignantAnalysis = false;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadOverview();
});

// 페이지 초기화
function initializePage() {
    // 로컬 스토리지에서 테마 불러오기
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        enableDarkMode();
    }
    
    // 악성 분석 체크박스 상태 불러오기
    const savedMalignant = localStorage.getItem('malignantAnalysis');
    if (savedMalignant === 'true') {
        document.getElementById('malignant-checkbox').checked = true;
        toggleMalignantAnalysis();
    }
}

// 테마 토글
function toggleTheme() {
    darkMode = !darkMode;
    if (darkMode) {
        enableDarkMode();
    } else {
        enableLightMode();
    }
}

function enableDarkMode() {
    document.body.classList.add('dark-mode');
    document.body.classList.remove('light-mode');
    document.querySelector('.main-content').classList.remove('light-mode');
    document.getElementById('theme-icon').textContent = '☀️';
    darkMode = true;
    localStorage.setItem('theme', 'dark');
}

function enableLightMode() {
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
    document.querySelector('.main-content').classList.add('light-mode');
    document.getElementById('theme-icon').textContent = '🌙';
    darkMode = false;
    localStorage.setItem('theme', 'light');
}

// 악성 분석 토글
function toggleMalignantAnalysis() {
    showMalignantAnalysis = document.getElementById('malignant-checkbox').checked;
    const menuItem = document.getElementById('malignant-menu-item');
    
    if (showMalignantAnalysis) {
        menuItem.classList.remove('hidden');
    } else {
        menuItem.classList.add('hidden');
        if (currentPage === 'malignant-analysis') {
            showPage('overview');
        }
    }
    
    localStorage.setItem('malignantAnalysis', showMalignantAnalysis);
}

// 페이지 전환
function showPage(pageName) {
    // 모든 페이지 숨기기
    document.querySelectorAll('.page-content').forEach(page => {
        page.classList.add('hidden');
    });
    
    // 선택한 페이지 표시
    document.getElementById(`page-${pageName}`).classList.remove('hidden');
    
    // 메뉴 활성화
    document.querySelectorAll('.sidebar-menu-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.sidebar-menu-item').classList.add('active');
    
    currentPage = pageName;
    
    // 페이지별 데이터 로드
    switch(pageName) {
        case 'overview':
            loadOverview();
            break;
        case 'visualization':
            loadVisualization();
            break;
        case 'ml-models':
            // 모델 페이지는 버튼 클릭 시 로드
            break;
        case 'dimension-reduction':
            // 차원 축소 페이지는 버튼 클릭 시 로드
            break;
        case 'correlation':
            // 상관관계 분석은 버튼 클릭 시 로드
            break;
        case 'malignant-analysis':
            // 악성 분석은 버튼 클릭 시 로드
            break;
    }
}

// 데이터 개요 로드
async function loadOverview() {
    try {
        // 개요 메트릭 로드
        const overviewRes = await fetch('/api/data/overview');
        const overviewData = await overviewRes.json();
        
        if (overviewData.error) {
            document.getElementById('overview-metrics').innerHTML = `<p>오류: ${overviewData.error}</p>`;
            return;
        }
        
        // 메트릭 카드 생성
        document.getElementById('overview-metrics').innerHTML = `
            <div class="metric-card">
                <label>총 샘플 수</label>
                <div class="value">${overviewData.total_samples}</div>
            </div>
            <div class="metric-card">
                <label>양성 (B) 샘플</label>
                <div class="value">${overviewData.benign_count}</div>
            </div>
            <div class="metric-card">
                <label>악성 (M) 샘플</label>
                <div class="value">${overviewData.malignant_count}</div>
            </div>
            <div class="metric-card">
                <label>악성 비율</label>
                <div class="value">${overviewData.malignant_pct}%</div>
            </div>
        `;
        
        // 진단 분포 로드
        const distRes = await fetch('/api/data/diagnosis-distribution');
        const distData = await distRes.json();
        
        // 파이 차트
        const pieData = [{
            values: distData.values,
            labels: distData.labels,
            type: 'pie',
            marker: {
                colors: distData.colors
            },
            textinfo: 'label+percent',
            textposition: 'inside'
        }];
        
        Plotly.newPlot('diagnosis-pie-chart', pieData, {
            title: '진단 분포',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        // 바 차트
        const barData = [{
            x: distData.labels,
            y: distData.values,
            type: 'bar',
            marker: {
                color: distData.colors
            }
        }];
        
        Plotly.newPlot('diagnosis-bar-chart', barData, {
            title: '진단별 샘플 수',
            xaxis: { title: '진단' },
            yaxis: { title: '샘플 수' },
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        // 데이터 미리보기 로드
        const previewRes = await fetch('/api/data/preview');
        const previewData = await previewRes.json();
        
        if (previewData.error) {
            document.getElementById('data-preview').innerHTML = `<p>오류: ${previewData.error}</p>`;
            return;
        }
        
        // 테이블 생성
        let tableHTML = '<table><thead><tr>';
        previewData.columns.forEach(col => {
            tableHTML += `<th>${col}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';
        
        previewData.data.forEach(row => {
            tableHTML += '<tr>';
            row.forEach(cell => {
                tableHTML += `<td>${typeof cell === 'number' ? cell.toFixed(2) : cell}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table>';
        
        document.getElementById('data-preview').innerHTML = tableHTML;
        
    } catch (error) {
        console.error('개요 데이터 로드 오류:', error);
        document.getElementById('overview-metrics').innerHTML = `<p>데이터 로드 오류: ${error.message}</p>`;
    }
}

// 시각화 페이지 로드
async function loadVisualization() {
    try {
        // 특징 목록 가져오기
        const featuresRes = await fetch('/api/features/list');
        const featuresData = await featuresRes.json();
        
        if (featuresData.error) {
            document.getElementById('feature-selector').innerHTML = `<p>오류: ${featuresData.error}</p>`;
            return;
        }
        
        // 특징 선택 UI 생성
        const features = featuresData.features.slice(0, 10);
        let checkboxesHTML = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">';
        features.forEach((feature, index) => {
            checkboxesHTML += `
                <label style="display: flex; align-items: center; gap: 0.25rem;">
                    <input type="checkbox" class="feature-checkbox" value="${feature}" ${index < 5 ? 'checked' : ''}>
                    <span>${feature}</span>
                </label>
            `;
        });
        checkboxesHTML += '</div>';
        checkboxesHTML += '<button class="btn" onclick="loadBoxplot()">Boxplot 생성</button>';
        
        document.getElementById('feature-selector').innerHTML = checkboxesHTML;
        
        // 히스토그램 특징 선택 UI
        let histCheckboxesHTML = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">';
        features.slice(0, 10).forEach((feature, index) => {
            histCheckboxesHTML += `
                <label style="display: flex; align-items: center; gap: 0.25rem;">
                    <input type="checkbox" class="hist-feature-checkbox" value="${feature}" ${index < 3 ? 'checked' : ''}>
                    <span>${feature}</span>
                </label>
            `;
        });
        histCheckboxesHTML += '</div>';
        histCheckboxesHTML += '<button class="btn" onclick="loadHistogram()">히스토그램 생성</button>';
        
        document.getElementById('histogram-feature-selector').innerHTML = histCheckboxesHTML;
        
        // 기본 Boxplot 로드
        loadBoxplot();
        loadHistogram();
        
    } catch (error) {
        console.error('시각화 페이지 로드 오류:', error);
    }
}

// Boxplot 로드
async function loadBoxplot() {
    const selectedFeatures = Array.from(document.querySelectorAll('.feature-checkbox:checked'))
        .map(cb => cb.value);
    
    if (selectedFeatures.length === 0) {
        document.getElementById('boxplot-chart').innerHTML = '<p>최소 1개 이상의 특징을 선택해주세요.</p>';
        return;
    }
    
    try {
        const params = new URLSearchParams();
        selectedFeatures.forEach(f => params.append('features', f));
        
        const res = await fetch(`/api/visualization/boxplot?${params.toString()}`);
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('boxplot-chart').innerHTML = `<p>오류: ${data.error}</p>`;
            return;
        }
        
        // Plotly boxplot 생성
        const traces = data.data.map(item => ({
            y: item.values,
            name: item.feature,
            type: 'box',
            boxmean: 'sd'
        }));
        
        Plotly.newPlot('boxplot-chart', traces, {
            title: '특징별 Boxplot',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
    } catch (error) {
        console.error('Boxplot 로드 오류:', error);
        document.getElementById('boxplot-chart').innerHTML = `<p>오류: ${error.message}</p>`;
    }
}

// 히스토그램 로드
async function loadHistogram() {
    const selectedFeatures = Array.from(document.querySelectorAll('.hist-feature-checkbox:checked'))
        .map(cb => cb.value);
    
    if (selectedFeatures.length === 0) {
        document.getElementById('histogram-chart').innerHTML = '<p>최소 1개 이상의 특징을 선택해주세요.</p>';
        return;
    }
    
    try {
        const params = new URLSearchParams();
        selectedFeatures.forEach(f => params.append('features', f));
        
        const res = await fetch(`/api/visualization/histogram?${params.toString()}`);
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('histogram-chart').innerHTML = `<p>오류: ${data.error}</p>`;
            return;
        }
        
        // 서브플롯 생성
        const rows = Math.ceil(Math.sqrt(selectedFeatures.length));
        const cols = Math.ceil(selectedFeatures.length / rows);
        
        const plots = [];
        const annotations = [];
        
        data.data.forEach((item, idx) => {
            const row = Math.floor(idx / cols) + 1;
            const col = (idx % cols) + 1;
            
            // 양성 데이터
            plots.push({
                x: item.benign,
                name: '양성 (B)',
                type: 'histogram',
                marker: { color: '#48BBB4' },
                opacity: 0.7,
                xaxis: `x${idx + 1 === 1 ? '' : idx + 1}`,
                yaxis: `y${idx + 1 === 1 ? '' : idx + 1}`,
                showlegend: idx === 0
            });
            
            // 악성 데이터
            plots.push({
                x: item.malignant,
                name: '악성 (M)',
                type: 'histogram',
                marker: { color: '#FF6B9D' },
                opacity: 0.7,
                xaxis: `x${idx + 1 === 1 ? '' : idx + 1}`,
                yaxis: `y${idx + 1 === 1 ? '' : idx + 1}`,
                showlegend: idx === 0
            });
        });
        
        const layout = {
            title: '진단별 특징 분포 비교',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            barmode: 'overlay',
            grid: { rows: rows, columns: cols, pattern: 'independent' }
        };
        
        Plotly.newPlot('histogram-chart', plots, layout, {responsive: true});
        
    } catch (error) {
        console.error('히스토그램 로드 오류:', error);
        document.getElementById('histogram-chart').innerHTML = `<p>오류: ${error.message}</p>`;
    }
}

// 머신러닝 모델 학습
async function trainModels() {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '학습 중...';
    
    try {
        const res = await fetch('/api/ml/train');
        const data = await res.json();
        
        if (data.error) {
            alert(`오류: ${data.error}`);
            return;
        }
        
        // 정확도 표시
        document.getElementById('rf-accuracy').textContent = `${(data.rf_accuracy * 100).toFixed(2)}%`;
        document.getElementById('dt-accuracy').textContent = `${(data.dt_accuracy * 100).toFixed(2)}%`;
        
        // Confusion Matrix
        Plotly.newPlot('rf-cm-chart', [{
            z: data.rf_cm,
            type: 'heatmap',
            colorscale: 'Blues',
            text: data.rf_cm.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: { size: 16 },
            showscale: false
        }], {
            xaxis: { title: '예측', tickvals: [0, 1], ticktext: ['양성 (B)', '악성 (M)'] },
            yaxis: { title: '실제', tickvals: [0, 1], ticktext: ['양성 (B)', '악성 (M)'] },
            title: 'Random Forest Confusion Matrix',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        Plotly.newPlot('dt-cm-chart', [{
            z: data.dt_cm,
            type: 'heatmap',
            colorscale: 'Oranges',
            text: data.dt_cm.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: { size: 16 },
            showscale: false
        }], {
            xaxis: { title: '예측', tickvals: [0, 1], ticktext: ['양성 (B)', '악성 (M)'] },
            yaxis: { title: '실제', tickvals: [0, 1], ticktext: ['양성 (B)', '악성 (M)'] },
            title: 'Decision Tree Confusion Matrix',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        // Feature Importance
        Plotly.newPlot('feature-importance-chart', [{
            x: data.feature_importance.importance,
            y: data.feature_importance.features,
            type: 'bar',
            orientation: 'h',
            marker: { color: data.feature_importance.importance, colorscale: 'Viridis' }
        }], {
            title: 'Feature Importance (상위 15개)',
            xaxis: { title: '중요도' },
            yaxis: { title: '특징' },
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        btn.disabled = false;
        btn.textContent = '모델 학습';
        alert('모델 학습이 완료되었습니다!');
        
    } catch (error) {
        console.error('모델 학습 오류:', error);
        alert(`오류: ${error.message}`);
        btn.disabled = false;
        btn.textContent = '모델 학습';
    }
}

// PCA 업데이트
async function updatePCA() {
    const nComponents = parseInt(document.getElementById('pca-components').value);
    
    try {
        const res = await fetch(`/api/dimension-reduction/pca?n_components=${nComponents}`);
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('pca-chart').innerHTML = `<p>오류: ${data.error}</p>`;
            return;
        }
        
        document.getElementById('pca-chart').innerHTML = `
            <p><strong>설명된 분산 비율:</strong> ${data.explained_variance.map(v => (v * 100).toFixed(2) + '%').join(', ')}</p>
            <p><strong>총 설명된 분산:</strong> ${(data.total_explained * 100).toFixed(2)}%</p>
        `;
        
        if (nComponents === 2) {
            // 2D 산점도
            const traces = [{
                x: data.data.map((d, i) => d[0]),
                y: data.data.map((d, i) => d[1]),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: data.labels,
                    colorscale: [[0, '#48BBB4'], [1, '#FF6B9D']],
                    size: 5,
                    opacity: 0.7
                },
                text: data.labels.map(l => l === 0 ? '양성 (B)' : '악성 (M)'),
                hovertemplate: '%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            }];
            
            Plotly.newPlot('pca-chart', traces, {
                title: `PCA ${nComponents}D 시각화`,
                xaxis: { title: `PC1 (${(data.explained_variance[0] * 100).toFixed(2)}%)` },
                yaxis: { title: `PC2 (${(data.explained_variance[1] * 100).toFixed(2)}%)` },
                font: { family: 'Malgun Gothic, sans-serif' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            }, {responsive: true});
        } else {
            // 3D 산점도
            const trace = {
                x: data.data.map(d => d[0]),
                y: data.data.map(d => d[1]),
                z: data.data.map(d => d[2]),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    color: data.labels,
                    colorscale: [[0, '#48BBB4'], [1, '#FF6B9D']],
                    size: 3,
                    opacity: 0.7
                },
                text: data.labels.map(l => l === 0 ? '양성 (B)' : '악성 (M)')
            };
            
            Plotly.newPlot('pca-chart', [trace], {
                title: `PCA ${nComponents}D 시각화`,
                scene: {
                    xaxis: { title: `PC1 (${(data.explained_variance[0] * 100).toFixed(2)}%)` },
                    yaxis: { title: `PC2 (${(data.explained_variance[1] * 100).toFixed(2)}%)` },
                    zaxis: { title: `PC3 (${(data.explained_variance[2] * 100).toFixed(2)}%)` }
                },
                font: { family: 'Malgun Gothic, sans-serif' },
                paper_bgcolor: 'rgba(0,0,0,0)'
            }, {responsive: true});
        }
        
    } catch (error) {
        console.error('PCA 오류:', error);
        document.getElementById('pca-chart').innerHTML = `<p>오류: ${error.message}</p>`;
    }
}

// t-SNE 업데이트
async function updateTSNE() {
    const nComponents = parseInt(document.getElementById('tsne-components').value);
    const perplexity = parseInt(document.getElementById('tsne-perplexity').value);
    
    document.getElementById('tsne-chart').innerHTML = '<div class="spinner"></div><p>t-SNE 계산 중... (시간이 걸릴 수 있습니다)</p>';
    
    try {
        const res = await fetch(`/api/dimension-reduction/tsne?n_components=${nComponents}&perplexity=${perplexity}`);
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('tsne-chart').innerHTML = `<p>오류: ${data.error}</p>`;
            return;
        }
        
        if (nComponents === 2) {
            const trace = {
                x: data.data.map(d => d[0]),
                y: data.data.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: data.labels,
                    colorscale: [[0, '#48BBB4'], [1, '#FF6B9D']],
                    size: 5,
                    opacity: 0.7
                },
                text: data.labels.map(l => l === 0 ? '양성 (B)' : '악성 (M)'),
                hovertemplate: '%{text}<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
            };
            
            Plotly.newPlot('tsne-chart', [trace], {
                title: 't-SNE 2D 시각화',
                xaxis: { title: 't-SNE 1' },
                yaxis: { title: 't-SNE 2' },
                font: { family: 'Malgun Gothic, sans-serif' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            }, {responsive: true});
        } else {
            const trace = {
                x: data.data.map(d => d[0]),
                y: data.data.map(d => d[1]),
                z: data.data.map(d => d[2]),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    color: data.labels,
                    colorscale: [[0, '#48BBB4'], [1, '#FF6B9D']],
                    size: 3,
                    opacity: 0.7
                },
                text: data.labels.map(l => l === 0 ? '양성 (B)' : '악성 (M)')
            };
            
            Plotly.newPlot('tsne-chart', [trace], {
                title: 't-SNE 3D 시각화',
                scene: {
                    xaxis: { title: 't-SNE 1' },
                    yaxis: { title: 't-SNE 2' },
                    zaxis: { title: 't-SNE 3' }
                },
                font: { family: 'Malgun Gothic, sans-serif' },
                paper_bgcolor: 'rgba(0,0,0,0)'
            }, {responsive: true});
        }
        
    } catch (error) {
        console.error('t-SNE 오류:', error);
        document.getElementById('tsne-chart').innerHTML = `<p>오류: ${error.message}</p>`;
    }
}

// 상관관계 행렬 로드
async function loadCorrelationMatrix() {
    document.getElementById('correlation-chart').innerHTML = '<div class="spinner"></div><p>상관관계 행렬 계산 중...</p>';
    
    try {
        const res = await fetch('/api/correlation/matrix');
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('correlation-chart').innerHTML = `<p>오류: ${data.error}</p>`;
            return;
        }
        
        Plotly.newPlot('correlation-chart', [{
            z: data.matrix,
            x: data.features,
            y: data.features,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0
        }], {
            title: '상관관계 히트맵',
            xaxis: { title: '특징', tickangle: -45 },
            yaxis: { title: '특징' },
            font: { family: 'Malgun Gothic, sans-serif', size: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
    } catch (error) {
        console.error('상관관계 행렬 오류:', error);
        document.getElementById('correlation-chart').innerHTML = `<p>오류: ${error.message}</p>`;
    }
}

// 악성 심각도 분석
async function analyzeMalignant() {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '분석 중...';
    
    document.getElementById('malignant-results').innerHTML = '<div class="spinner"></div><p>악성 심각도 분석 중...</p>';
    
    try {
        const res = await fetch('/api/malignant/analyze');
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('malignant-results').innerHTML = `<p>오류: ${data.error}</p>`;
            btn.disabled = false;
            btn.textContent = '악성 심각도 분석 실행';
            return;
        }
        
        let resultsHTML = `
            <h3>📊 악성 심각도 분포</h3>
            <div class="metric-grid" style="margin: 1rem 0;">
                <div class="metric-card">
                    <label>저악성 (0)</label>
                    <div class="value">${data.low_severity_count}</div>
                </div>
                <div class="metric-card">
                    <label>고악성 (1)</label>
                    <div class="value">${data.high_severity_count}</div>
                </div>
                <div class="metric-card">
                    <label>고악성 비율</label>
                    <div class="value">${data.high_severity_pct}%</div>
                </div>
                <div class="metric-card">
                    <label>모델 정확도</label>
                    <div class="value">${(data.model_accuracy * 100).toFixed(2)}%</div>
                </div>
            </div>
            
            <div class="chart-container" style="margin-top: 2rem;">
                <h3>악성 심각도 분포 (파이 차트)</h3>
                <div id="malignant-severity-pie"></div>
            </div>
            
            <div class="chart-container" style="margin-top: 2rem;">
                <h3>혼동행렬</h3>
                <div id="malignant-cm"></div>
            </div>
            
            <div class="chart-container" style="margin-top: 2rem;">
                <h3>악성 심각도 예측에 중요한 특징 (상위 15개)</h3>
                <div id="malignant-importance"></div>
            </div>
        `;
        
        document.getElementById('malignant-results').innerHTML = resultsHTML;
        
        // 파이 차트
        Plotly.newPlot('malignant-severity-pie', [{
            values: [data.low_severity_count, data.high_severity_count],
            labels: ['저악성 (0)', '고악성 (1)'],
            type: 'pie',
            marker: { colors: ['#48BBB4', '#FF6B9D'] },
            textinfo: 'label+percent',
            textposition: 'inside'
        }], {
            title: '악성 심각도 분포',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        // Confusion Matrix
        Plotly.newPlot('malignant-cm', [{
            z: data.confusion_matrix,
            type: 'heatmap',
            colorscale: 'Blues',
            text: data.confusion_matrix.map(row => row.map(val => val.toString())),
            texttemplate: '%{text}',
            textfont: { size: 16 },
            showscale: false
        }], {
            xaxis: { title: '예측', tickvals: [0, 1], ticktext: ['저악성', '고악성'] },
            yaxis: { title: '실제', tickvals: [0, 1], ticktext: ['저악성', '고악성'] },
            title: '혼동행렬',
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        // Feature Importance
        Plotly.newPlot('malignant-importance', [{
            x: data.feature_importance.importance,
            y: data.feature_importance.features,
            type: 'bar',
            orientation: 'h',
            marker: { color: data.feature_importance.importance, colorscale: 'Reds' }
        }], {
            title: '악성 심각도 예측에 중요한 특징',
            xaxis: { title: '중요도' },
            yaxis: { title: '특징' },
            font: { family: 'Malgun Gothic, sans-serif' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }, {responsive: true});
        
        btn.disabled = false;
        btn.textContent = '악성 심각도 분석 실행';
        
    } catch (error) {
        console.error('악성 분석 오류:', error);
        document.getElementById('malignant-results').innerHTML = `<p>오류: ${error.message}</p>`;
        btn.disabled = false;
        btn.textContent = '악성 심각도 분석 실행';
    }
}
