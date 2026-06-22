#  PitWall-AI

> **FastF1 레이스 데이터 · LLM 에이전트 · RAG를 결합한 Formula 1 분석 도구**

사용자가 시즌 / 그랑프리 / 드라이버를 선택하면 **레이스 브리핑, 전략 분석, 텔레메트리 비교, 피트스탑 시뮬레이션**을 한 화면에서 제공합니다. 정형 데이터(FastF1)는 SQLite로, 뉴스·규정 같은 비정형 데이터는 MongoDB·Qdrant 기반 RAG로 다룹니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/Agent-LlamaIndex-black)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-DC244C)

---

##  아키텍처

Streamlit UI 아래에 **4개의 LlamaIndex ReActAgent**가 동작하며, 데이터는 목적에 따라 3개 계층으로 분리됩니다.

```
Streamlit App
  │
  ├─ Briefing Agent ────── SQLite(race_results) 직접 조회 → 프롬프트 주입
  │                        + Qdrant RAG(뉴스 / 인터뷰 / FIA 규정) 보조
  │
  ├─ Strategy Agent ────── FastF1 세션 → 스틴트 / 타이어 / 트래픽 / 피트 분석
  │                        + Text2SQL(SQLite) 기록 조회
  │
  ├─ Analyst Agent ─────── FastF1 텔레메트리 → 랩타임 / 도미넌스 / 스피드 트레이스 시각화
  │
  └─ Tactical Sim Agent ── 언더컷 / 오버컷 피트 전술 시뮬레이션
```

| 계층 | 저장소 | 용도 |
|---|---|---|
| 정형 (Structured) | SQLite (`f1_data.db`) | 레이스 결과 · 랩타임 · 날씨 (FastF1 적재) |
| 비정형 원문 (Document) | MongoDB (Atlas) | 크롤링한 기사 · FIA 규정 원문 |
| 벡터 (Vector) | Qdrant (Cloud) | 임베딩 인덱스 (RAG 검색) |

---

##  에이전트별 기능

### 1. Briefing Agent · `gemini-2.5-pro`
- 선택한 연도·그랑프리의 공식 결과를 **SQLite에서 직접 조회**한 뒤 프롬프트에 주입(Data Injection)합니다.
- 순위·드라이버·팀명 같은 핵심 팩트는 **DB 값만 사용**하며, 약어→풀네임 변환도 파이썬이 직접 처리해 LLM의 환각·순위 오류를 차단합니다.
- 사건 원인·인터뷰·규정 근거가 필요할 때만 RAG 도구를 보조로 호출합니다. 검색 실패 시 추측 없이 "확인되지 않았습니다"로 보고하도록 강제합니다.

### 2. Strategy Agent · `gemini-2.0-flash`
- FastF1 세션 데이터로 **스틴트별 페이스, 타이어 마모(degradation), 트래픽 비율, 피트 타이밍**을 분석합니다.
- 전체 필드의 타이어 수명 통계를 기준으로 스틴트 길이를 평가(Short / Standard / Long / Extreme)합니다.
- 결과는 JSON 스키마로 출력되어 Streamlit 테이블로 렌더링됩니다.

### 3. Analyst Agent · `gemini-2.0-flash-exp`
- 두 드라이버의 **랩타임 비교(Plotly)**, **트랙 도미넌스 맵(Matplotlib)**, **스피드 트레이스(Plotly)**를 생성합니다.
- 의도에 맞는 시각화 도구 하나를 선택해 실행하고, 생성된 차트에 분석 코멘트를 덧붙입니다.

### 4. Tactical Simulation Agent · `gemini-2.5-pro`
- 피트스탑 타이밍을 기반으로 **언더컷 성공 확률, 스틴트 연장 손익**을 시뮬레이션합니다.
- FastF1에 데이터가 있는 과거 시즌은 실측 랩타임 기반으로 계산하며, 데이터가 없는 최신/미래 시즌은 가정값(피트 로스, 타이어 델타) 기반 추정으로 처리합니다.

---

##  RAG 파이프라인

- **원문 수집:** Autosport · Formula1.com 기사 + FIA 기술/스포팅 규정 PDF → MongoDB(Beanie ODM / Motor) 저장
- **임베딩:** Google Gemini Embedding (`gemini-embedding-001`, 3072차원)
- **벡터 스토어:** Qdrant (COSINE 거리), `platform` 메타데이터 필드 인덱싱으로 출처 필터링 지원
- **검색:** dense 벡터 시맨틱 검색 (`score_threshold` 기반 필터링) — `F1Retriever`가 단일 쿼리 임베딩으로 Qdrant를 조회

---

##  데이터 파이프라인

### Airflow DAG (`pitwall_daily_pipeline`)
```
crawl_f1_official  ──▶  crawl_autosport  ──▶  rag_indexing
   (Formula1.com)        (Autosport)          (MongoDB → Qdrant)
```
- 14일 주기 실행, `max_active_runs=1`로 중복 실행 방지
- **크롤러:** Selenium Remote(`selenium-chrome` 컨테이너) + trafilatura / BeautifulSoup 본문 추출
- **FIA 규정:** `requests` + `pypdf`로 PDF 파싱 후 MongoDB에 Upsert (URL 기준 멱등성)
- **인덱서(`RAGIndexer`):** MongoDB → Gemini 임베딩 → Qdrant Upsert, URL 기반 `uuid5`로 멱등성 보장, 배치 처리

### FastF1 Hard Data 적재
- `update_db.py` / `init_historical.py`가 FastF1 세션을 SQLite의 `race_results` · `lap_times` · `weather_data` 테이블에 적재
- 캐시는 `data/cache`를 사용하며, 읽기 전용 환경(예: Streamlit Cloud)에서는 `/tmp`로 폴백


---

##  프로젝트 구조

```
PitWall-AI/
├── app/
│   ├── agents/
│   │   ├── briefing_agent.py        # 레이스 브리핑 (Data Injection + RAG)
│   │   ├── strategy_agent.py        # 전략 감사 (스틴트/타이어/트래픽)
│   │   ├── analytic_agent.py        # 텔레메트리 시각화
│   │   └── tactic_simulation_agent.py  # 언더컷/오버컷 시뮬
│   ├── tools/
│   │   ├── deterministic_data.py    # SQLite race_results 직접 조회
│   │   ├── hard_data.py             # Text2SQL (NLSQLTableQueryEngine)
│   │   ├── soft_data.py             # RAG 통합 검색
│   │   └── telemetry_data.py        # FastF1 플롯 생성
│   └── regulation_tool.py           # FIA 규정 RAG 도구
│
├── data_pipeline/
│   ├── crawlers/                    # Selenium 크롤러 + FIA PDF 크롤러
│   ├── pipelines/                   # FastF1 → SQLite 적재 스크립트
│   ├── analytics.py                 # 전략/타이어 분석 엔진
│   ├── rag_indexer.py               # MongoDB → Qdrant 인덱싱
│   └── retriever.py                 # Qdrant 벡터 검색
│
├── dags/pitwall_pipeline.py         # Airflow DAG
├── domain/documents.py              # MongoDB 스키마 (Beanie)
├── data/f1_data.db                  # SQLite DB
├── streamlit_app.py
├── docker-compose.yaml
├── Dockerfile  ·  Dockerfile.airflow
├── requirements.txt  ·  requirements_airflow.txt
└── README.md
```

---

##  환경 변수

`.env` 파일에 아래 값을 설정합니다.

```env
GOOGLE_API_KEY=your_google_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
MONGO_URI=your_mongodb_uri
AIRFLOW_FERNET_KEY=your_airflow_fernet_key
AIRFLOW_UID=50000
```

---

##  실행

### Streamlit 로컬 실행
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 전체 스택 실행 (Docker Compose)
```bash
docker-compose up -d
docker-compose run airflow-init
```

| 서비스 | URL |
|---|---|
| Streamlit App | http://localhost:8501 |
| Airflow UI | http://localhost:8080 |
| Mongo Express | http://localhost:8081 |
| Qdrant | http://localhost:6333 |

### 데이터 상태 확인
```bash
python data_test.py   # MongoDB · Qdrant · SQLite 적재 상태 점검
```

---

##  배포

- **App:** Streamlit Community Cloud
- **Managed Services:** MongoDB Atlas, Qdrant Cloud

---

##  라이선스

Personal / educational project. F1 데이터는 FastF1을 통해 수집됩니다.