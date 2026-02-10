# PitWall-AI



## FastF1 라이브러리, RAG System을 활용한 F1 정보 요약 및 가이드 에이전트



# 🏎️ PitWall-AI: Formula 1 RAG Intelligence System

> **"Your Personal Race Engineer Powered by LLM"**
> 복잡한 F1 규정(Sporting, Technical, Financial)과 레이스 데이터를 LLM이 분석하여, 팬들에게 실시간으로 답변해주는 AI 서비스입니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-black)
![GCP](https://img.shields.io/badge/Deploy-Cloud%20Run-4285F4?logo=google-cloud)

## 🏗️ Architecture
- **Frontend:** Streamlit
- **LLM Engine:** LlamaIndex (Agentic Workflow)
- **Model:** Google Gemini Pro (Reasoning), BAAI/bge-m3 (Embedding)
- **Vector DB:** Qdrant Cloud (Hybrid Search)
- **Infrastructure:** Google Cloud Platform (Cloud Build, Artifact Registry, Cloud Run)
- **Containerization:** Docker

## ✨ Key Features
1.  **Regulation Expert:** 수백 페이지의 FIA 규정집(PDF)을 RAG로 검색하여 정확한 근거와 조항을 제시.
2.  **Context-Aware Chat:** 이전 대화 맥락을 기억하는 멀티턴(Multi-turn) 대화 지원.
3.  **Hybrid Search:** 키워드 매칭(Sparse)과 의미 기반 검색(Dense)을 결합하여 검색 정확도 향상.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker & Google Cloud CLI
- API Keys (Google Gemini, Qdrant)

### 1. Installation
```bash
git clone [https://github.com/your-username/PitWall-AI.git](https://github.com/your-username/PitWall-AI.git)
cd PitWall-AI
pip install -r requirements.txt
