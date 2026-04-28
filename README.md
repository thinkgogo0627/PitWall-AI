# 🏎️ PitWall-AI

> **"Your Personal Race Engineer Powered by LLM"**
> FastF1 텔레메트리, 멀티 에이전트 AI, 벡터 DB 기반 RAG를 결합한 Formula 1 종합 분석 시스템

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-black)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-EF2C6B)
![Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-017CEE?logo=apache-airflow)

---

## 📋 목차

1. [프로젝트 소개](#-프로젝트-소개)
2. [아키텍처](#-아키텍처)
3. [주요 기능](#-주요-기능)
4. [기술 스택](#-기술-스택)
5. [프로젝트 구조](#-프로젝트-구조)
6. [시작하기](#-시작하기)
7. [데이터 파이프라인](#-데이터-파이프라인)

---

## 🏁 프로젝트 소개

PitWall-AI는 F1 팬과 분석가를 위한 **AI 레이스 엔지니어** 시스템입니다.

- **레이스 브리핑:** 경기 결과, 드라이버별 분석, 주요 사건 정리
- **전략 분석:** 스틴트별 타이어 전략, 페이스 비교, 언더컷/오버컷 시뮬레이션
- **텔레메트리 시각화:** FastF1 기반 드라이버 랩타임, 속도 트레이스, 트랙 지배도 분석
- **FIA 규정 검색:** 수백 페이지 규정집(PDF)을 RAG로 검색하여 정확한 조항 근거 제시
- **F1 뉴스 컨텍스트:** Autosport, Formula1.com 크롤링 기반 최신 정보 검색

---

## 🏗️ 아키텍처

```
                        ┌─────────────────────┐
                        │   Streamlit Frontend │
                        │  (Briefing / Tele /  │
                        │    Strategy Tabs)    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       LlamaIndex ReActAgent  │
                    │  (Briefing / Strategy /      │
                    │   Simulation / Analytic)     │
                    └──────┬──────────────┬────────┘
                           │              │
            ┌──────────────▼──┐   ┌───────▼──────────────┐
            │   Hard Data     │   │     Soft Data         │
            │  (Text2SQL)     │   │  (RAG / Qdrant)       │
            │  SQLite DB      │   │  Google Gemini Embed  │
            └─────────────────┘   └───────────────────────┘
                           │              │
            ┌──────────────▼──┐   ┌───────▼──────────────┐
            │  FastF1 API     │   │   MongoDB Atlas       │
            │  (Telemetry)    │   │  (Document Storage)   │
            └─────────────────┘   └───────────────────────┘
                                           │
                              ┌────────────▼───────────────┐
                              │  Airflow DAG (14일 주기)    │
                              │  Selenium Crawler           │
                              │  (Autosport + Formula1.com) │
                              └────────────────────────────┘
```

---

## ✨ 주요 기능

### 1. 레이스 브리핑 에이전트
- 경기 전체 요약 또는 드라이버별 심층 분석 모드 전환
- SQLite 레이스 결과 DB + 뉴스/인터뷰 컨텍스트 결합
- FIA 규정 조항 자동 참조 (패널티, 사건 분석 시)

### 2. 전략 분석 에이전트
- Text2SQL 기반 레이스 데이터 쿼리 (랩타임, 날씨, 피트스탑)
- 스틴트 평가: Long Run / Normal / Short Stint 자동 분류
- 타이어 degradation 분석 + 트래픽 감지 (1.0초 갭 임계값)

### 3. 전술 시뮬레이션 에이전트
- 언더컷/오버컷 시나리오 실시간 시뮬레이션
- 피트 로스 계산 및 최적 피트 윈도우 추천
- FastF1 실제 텔레메트리 기반 타이어 모델링

### 4. 텔레메트리 분석
- 드라이버 랩타임 비교 차트
- 트랙 구간별 지배도 맵 (Track Dominance)
- 속도/스로틀/브레이크 트레이스 오버레이

### 5. RAG 검색 (Soft Data)
- Qdrant 벡터 DB 하이브리드 검색 (Dense + 메타데이터 필터링)
- Google Gemini Embedding (`models/gemini-embedding-001`, 3072차원)
- Autosport / Formula1.com 크롤링 아티클 색인

### 6. FIA 규정 검색
- FIA 공식 PDF(스포팅/테크니컬/파이낸셜 규정) 파싱 및 벡터화
- 규정 조항 정확 근거 제시

---

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **LLM** | Google Gemini 2.5 Pro / 2.5 Flash / 2.0 Flash |
| **Embedding** | Google Gemini Embedding (`gemini-embedding-001`) |
| **Agent Framework** | LlamaIndex (ReActAgent, FunctionTool) |
| **Vector DB** | Qdrant Cloud (Hybrid Search) |
| **Relational DB** | SQLite (`f1_data.db`) |
| **Document DB** | MongoDB Atlas (Beanie ODM) |
| **F1 Data** | FastF1 (공식 텔레메트리 API) |
| **Frontend** | Streamlit |
| **Orchestration** | Apache Airflow 2.9.1 |
| **Web Scraping** | Selenium + BeautifulSoup4 + Trafilatura |
| **Containerization** | Docker & Docker Compose |
| **Visualization** | Plotly, Matplotlib, FastF1 Plotting |
| **Async** | asyncio, Motor (MongoDB async driver) |
| **Data Validation** | Pydantic, Beanie |

---

## 📁 프로젝트 구조

```
PitWall-AI/
├── app/
│   ├── agents/
│   │   ├── briefing_agent.py       # 레이스 브리핑 에이전트
│   │   ├── strategy_agent.py       # 전략 분석 에이전트
│   │   ├── tactic_simulation_agent.py  # 전술 시뮬레이션 에이전트
│   │   └── analytic_agent.py       # 분석 에이전트
│   ├── tools/
│   │   ├── hard_data.py            # Text2SQL (SQLite 레이스 DB)
│   │   ├── soft_data.py            # RAG 검색 (Qdrant)
│   │   ├── deterministic_data.py   # 레이스 결과/순위 조회
│   │   └── telemetry_data.py       # FastF1 텔레메트리 시각화
│   └── regulation_tool.py          # FIA 규정 검색 툴
│
├── data_pipeline/
│   ├── crawlers/
│   │   ├── base.py                 # 크롤러 베이스 클래스
│   │   ├── f1_news.py              # Autosport 뉴스 크롤러
│   │   ├── f1_tactic.py            # Formula1.com 크롤러
│   │   └── FIA_reg.py              # FIA 규정 PDF 크롤러
│   ├── pipelines/
│   │   ├── init_historical.py      # 과거 데이터 초기화
│   │   ├── init_static.py          # 정적 데이터 초기화
│   │   └── update_db.py            # DB 업데이트
│   ├── analytics.py                # 레이스 전략 분석 엔진
│   ├── rag_indexer.py              # Qdrant 벡터 인덱싱
│   ├── retriever.py                # Qdrant 검색 엔진
│   └── update_db.py                # DB 업데이트 유틸
│
├── dags/
│   ├── pitwall_pipeline.py         # 메인 데이터 파이프라인 DAG
│   └── get_text2sql_dataset.py     # Text2SQL 데이터셋 생성 DAG
│
├── domain/
│   └── documents.py                # Beanie ODM 스키마
│
├── data/
│   ├── f1_data.db                  # SQLite 레이스 데이터
│   └── cache/                      # FastF1 캐시
│
├── streamlit_app.py                # Streamlit 메인 앱
├── docker-compose.yaml             # 멀티 컨테이너 구성
├── Dockerfile                      # Streamlit 컨테이너
├── Dockerfile.airflow              # Airflow 컨테이너
├── requirements.txt                # Python 의존성
└── requirements_airflow.txt        # Airflow 의존성
```

---

## 🚀 시작하기

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API Keys: Google Gemini, Qdrant Cloud, MongoDB Atlas

### 환경 변수 설정

`.env` 파일을 생성하고 아래 값을 입력합니다:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
MONGO_URI=your_mongodb_atlas_connection_string
AIRFLOW_FERNET_KEY=your_fernet_key
AIRFLOW_UID=50000
```

### 로컬 실행 (Streamlit만)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Streamlit 앱 실행
streamlit run streamlit_app.py

# 3. 데이터 상태 확인
python data_test.py
```

### Docker 전체 스택 실행 (권장)

```bash
# 전체 서비스 시작
docker-compose up -d

# Airflow 초기화 (최초 1회)
docker-compose run airflow-init
```

| 서비스 | URL |
|--------|-----|
| Streamlit App | http://localhost:8501 |
| Airflow UI | http://localhost:8080 |
| Mongo Express | http://localhost:8081 |
| Qdrant Dashboard | http://localhost:6333 |

---

## 🔄 데이터 파이프라인

Airflow DAG가 **14일 주기**로 자동 실행됩니다.

```
[Autosport 크롤러] ──┐
                     ├──→ [MongoDB Atlas] ──→ [RAG 인덱싱] ──→ [Qdrant]
[Formula1.com 크롤러]┘
```

1. **크롤링 (병렬):** Selenium Chrome으로 Autosport, Formula1.com 최신 기사 수집
2. **문서 저장:** MongoDB Atlas에 `F1NewsDocument` 스키마로 저장
3. **벡터 인덱싱:** Google Gemini API로 임베딩 후 Qdrant 업서트 (멱등성 보장)

### 수동 파이프라인 실행

```bash
# Airflow DAG 직접 트리거
airflow dags trigger pitwall_daily_pipeline

# 또는 Python으로 직접 실행
python data_pipeline/rag_indexer.py
```

---

## 📊 에이전트 도구 구성

| 에이전트 | 사용 도구 |
|---------|----------|
| **Briefing** | 레이스 결과 DB, FIA 규정 검색, 뉴스/인터뷰 RAG, 이벤트 타임라인, 웹 검색 |
| **Strategy** | Text2SQL (레이스 DB), 스틴트 분석, 타이어 degradation 분석 |
| **Simulation** | FastF1 텔레메트리, 피트 로스 계산, 언더컷/오버컷 시뮬레이션 |
| **Analytic** | 레이스 전반 통계 분석 |

---

## 📝 라이선스

This project is for personal/educational use. F1 data sourced via [FastF1](https://github.com/theOehrly/Fast-F1) (MIT License).
