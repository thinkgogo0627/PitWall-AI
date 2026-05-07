# PitWall-AI

FastF1 레이스 데이터, Streamlit UI, LLM 에이전트, Qdrant 기반 RAG를 결합한 Formula 1 레이스 분석 도구입니다.

주요 목적은 사용자가 특정 시즌/그랑프리/드라이버를 선택하면 레이스 결과, 스틴트 전략, 타이어 마모, 텔레메트리 비교, 뉴스/규정 근거를 한 화면에서 확인할 수 있게 하는 것입니다.

## 현재 동작 흐름

```
Streamlit App
  |
  |-- Briefing
  |     |-- SQLite race_results에서 공식 결과 조회
  |     |-- 조회된 표를 LLM 프롬프트에 직접 주입
  |     |-- 필요 시 Qdrant RAG로 뉴스/인터뷰/규정 컨텍스트 보강
  |
  |-- Strategy Center
  |     |-- FastF1 캐시/세션에서 랩, 스틴트, 타이어 데이터 로드
  |     |-- 트래픽, 클린 페이스, 스틴트 길이, 타이어 degradation 분석
  |
  |-- Telemetry / Visualization
  |     |-- FastF1 기반 랩타임 비교
  |     |-- 스피드 트레이스
  |     |-- 트랙 구간별 dominance plot
  |
  |-- Tactical Simulation
        |-- 피트스탑, 언더컷/오버컷 시나리오 분석
```

## 주요 기능

### Race Briefing

- 선택한 연도와 그랑프리의 공식 레이스 결과를 SQLite DB에서 조회합니다.
- 전체 레이스 요약과 드라이버별 Focus Report를 생성합니다.
- 드라이버명/순위/팀명 같은 핵심 팩트는 조회된 결과표를 기준으로 사용합니다.
- 뉴스, 인터뷰, FIA 규정 근거가 필요할 때 RAG 검색을 보조로 사용합니다.

### Strategy Center

- FastF1 캐시 데이터를 이용해 레이스 스틴트 정보를 불러옵니다.
- 드라이버별 타이어 컴파운드, 스틴트 길이, 트래픽 비율, 클린 페이스를 분석합니다.
- 서킷 전체의 타이어 평균 수명과 degradation 경향을 계산합니다.
- 결과는 Streamlit UI에서 JSON 기반 테이블 형태로 표시됩니다.

### Telemetry Analysis

- 두 드라이버의 Race Pace를 Plotly 차트로 비교합니다.
- Fastest Lap 기준 Speed Trace를 표시합니다.
- 트랙 좌표와 속도 차이를 이용해 어느 드라이버가 어느 구간에서 우세했는지 시각화합니다.

### RAG Context

- Autosport, Formula1.com, FIA 문서 등 비정형 텍스트를 MongoDB에 저장합니다.
- Gemini Embedding으로 벡터화한 뒤 Qdrant에 업서트합니다.
- 브리핑/규정 확인/사건 원인 설명에 필요한 보조 컨텍스트로 사용합니다.

### Data Pipeline

- Airflow DAG가 크롤러와 RAG 인덱서를 실행합니다.
- Selenium 기반 크롤러가 기사 목록과 본문을 수집합니다.
- MongoDB 저장 후 Qdrant 컬렉션에 벡터 인덱싱합니다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| App UI | Streamlit |
| F1 데이터 | FastF1, SQLite |
| LLM | Google Gemini |
| Agent | LlamaIndex ReActAgent, FunctionTool |
| Embedding | Google Gemini Embedding |
| Vector DB | Qdrant |
| Document DB | MongoDB / Motor / Beanie |
| Pipeline | Airflow, Selenium |
| Visualization | Plotly, Matplotlib |
| Runtime | Python 3.11, Docker Compose |

## 프로젝트 구조

```
PitWall-AI/
├── app/
│   ├── agents/
│   │   ├── briefing_agent.py
│   │   ├── strategy_agent.py
│   │   └── tactic_simulation_agent.py
│   ├── tools/
│   │   ├── deterministic_data.py
│   │   ├── soft_data.py
│   │   └── telemetry_data.py
│   └── regulation_tool.py
│
├── data_pipeline/
│   ├── crawlers/
│   ├── analytics.py
│   ├── rag_indexer.py
│   └── retriever.py
│
├── dags/
│   └── pitwall_pipeline.py
│
├── domain/
│   └── documents.py
│
├── data/
│   ├── f1_data.db
│   └── cache/
│
├── streamlit_app.py
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
└── requirements_airflow.txt
```

## 환경 변수

`.env` 파일에 아래 값을 설정합니다.

```env
GOOGLE_API_KEY=your_google_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
MONGO_URI=your_mongodb_uri
AIRFLOW_FERNET_KEY=your_airflow_fernet_key
AIRFLOW_UID=50000
```

## 실행

### Streamlit 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 전체 스택 실행

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

## 데이터 상태 확인

```bash
python data_test.py
```

이 스크립트는 MongoDB, Qdrant, SQLite 적재 상태를 확인합니다.

## 라이선스

Personal/educational project. F1 data is sourced through FastF1.
