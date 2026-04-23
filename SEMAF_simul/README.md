# SEMAF 실험 시뮬레이션 — 배포 가이드
<img width="1027" height="961" alt="image" src="https://github.com/user-attachments/assets/0b08ce34-5eab-4406-b2da-bf115616bc7b" />

**자기진화형 다중 에이전트 프레임워크(SEMAF)**  

---

## 📦 설치 및 로컬 실행

1) SEMAF_simul 폴더로 이동
2) 가상 환경 생성:
  $ python -m venv venv

3) 가상 환경 활성화: 
  $ venv\Scripts\activate

4) 의존성 설치
```bash
pip install -r requirements.txt
```

5) 로컬 실행
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 접속

---

## ☁️ 배포 옵션

### A. Streamlit Community Cloud (무료, 권장)

1. GitHub 리포지토리에 코드 업로드
   ```
   semaf_app/
   ├── app.py
   ├── requirements.txt
   └── .streamlit/
       └── config.toml
   ```

2. [share.streamlit.io](https://share.streamlit.io) 접속 → "New app"

3. 연결 설정:
   - **Repository**: `your-github/semaf-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. "Deploy" 클릭 → 자동 배포 완료

5. OpenAI API Key 보안 설정 (Streamlit Cloud):
   - App Settings → Secrets:
   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```

### B. Docker 배포

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t semaf-app .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-proj-... semaf-app
```

### C. Hugging Face Spaces (무료)

1. [huggingface.co/spaces](https://huggingface.co/spaces) → "Create new Space"
2. SDK: **Streamlit** 선택
3. 파일 업로드 후 자동 배포

---

## 🔧 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | (없으면 시뮬레이션 모드) |
| `STREAMLIT_SERVER_PORT` | 서버 포트 | 8501 |

---

## 📊 실험 파라미터 설명

| 파라미터 | 논문 기본값 | 설명 |
|----------|-------------|------|
| NUM_ITERATIONS | 60 (→100) | 총 시뮬레이션 반복 횟수 |
| NUM_AGENTS | 3 | 에이전트 수 |
| ADAPTATION_POINTS | [20, 50] | 환경 변화 측정 지점 |
| λQ | 1.0 | 품질 보상 가중치 |
| λC | 0.01 | 통신 비용 패널티 가중치 |
| λK | 0.5 | KRI 보상 가중치 |
| KRI_target | 0.95 | KRI 목표값 (95%) |

---

## 📐 메트릭 공식 (논문 4장)

```
LRA = (1/N) × Σ(1/Ti)         # 적응 학습률
CE  = Q / C                    # 협업 효율성
KRI = 1 - (D_new / D_total)   # 지식 보유 지수
R   = λQ·Q - λC·C + λK·(KRI - KRI_target)  # 보상 함수
```

---

## 📈 논문 Table 9 기준 결과

| 지표 | Baseline | SEMAF | 개선 |
|------|----------|-------|------|
| LRA | 0.0611 | 0.2500 | 4.09배 ↑ |
| CE  | 0.0377 | 0.1475 | 3.91배 ↑ |
| KRI | 0.8047 | 0.9509 | +14.62%p ↑ |

---

## 📁 파일 구조

```
semaf_app/
├── app.py                  # 메인 Streamlit 앱 (실험 UI + 시뮬레이션 엔진)
├── requirements.txt        # Python 의존성
├── README.md               # 이 파일
└── .streamlit/
    └── config.toml         # Streamlit 서버·테마 설정
```


