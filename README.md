# PitWall-AI



## FastF1 라이브러리, RAG System을 활용한 F1 정보 요약 및 가이드 에이전트



### 1. 시스템 아키텍처

- 데이터 흐름 수집 -> 저장 -> 서빙 단계로 설계함


### A. 데이터 파이프라인
- 정형 데이터 (Hard Data, tabular)
    - Source: FastF1 라이브러리 (공식 F1 레이스 데이터 제공)
    - Data: 랩타임, 타이어 컴파운드, 섹터 기록, 텔레메트리 (속도, 엔진 RPM)
    - 저장: SQLite
    - 파이프라인: 매 그랑프리 종료 시점마다 데이터 갱신 -> 현 시점까지 종료된 경기 데이터 전원 DB 적재(data/f1_data.db), 이후 그랑프리 시점마다 Airflow로 자동화



- 비정형 데이터 (Soft Data, text)
    - Source: GPKorea , autosport, F1 공식 홈페이지의 race strategy 섹션, FIA 공식 문서(F1 Technical Regulations, F1 Sporting Regulations)
    - Data: 위 데이터 소스들에 있는 문서들 -> (선수들의 경기 기록, 기술 규정, 기타 등등)
    - 저장: BAAI 모델 활용하여 임베딩 변환 후 ChromaDB에 적재
