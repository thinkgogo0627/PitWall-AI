# data/soft_data.py

## RAG 검색 도구 구현
## 

# data/soft_data.py

import sys
import os
import logging
from duckduckgo_search import DDGS

# [경로 설정] 로컬/Docker 어디서든 모듈을 찾을 수 있게
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# ✅ 우리가 방금 고친 그 엔진 임포트
from data_pipeline.retriever import F1Retriever

# 로거 설정
logger = logging.getLogger(__name__)

# --- 1. 검색 엔진 시동 (Global Instance) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

print(f" [SoftData] Connecting to Qdrant at {QDRANT_URL}...")

try:
    # F1Retriever 인스턴스 생성 (여기서 임베딩 모델 로드됨)
    retriever_engine = F1Retriever(qdrant_url=QDRANT_URL)
    print(" RAG Search Engine Ready.")
except Exception as e:
    print(f" RAG Engine Load Failed: {e}")
    retriever_engine = None


# ---------------------------------------------------------
# 🛠️ Helper: 검색 결과 포맷팅 (LLM이 읽기 좋게)
# ---------------------------------------------------------
def _format_rag_results(results: list) -> str:
    if not results:
        return "관련 정보를 찾지 못했습니다."
    
    context_list = []
    for i, hit in enumerate(results, 1):
        # retriever.search()가 반환하는 dict 구조 활용
        score = hit.get('score', 0.0)
        title = hit.get('title', 'No Title')
        source = hit.get('platform', 'Unknown Source') # platform 필드 사용
        date = hit.get('published_at', '')[:10]
        text = hit.get('text', '').strip()
        
        # 텍스트가 너무 길면 500자에서 자르기 (토큰 절약)
        if len(text) > 500:
            text = text[:500] + "...(more)"

        context_list.append(
            f"[{i}] 제목: {title}\n"
            f"    출처: {source} ({date}) | 유사도: {score:.3f}\n"
            f"    내용: {text}"
        )
    return "\n\n".join(context_list)


# ---------------------------------------------------------
#  1. 드라이버 인터뷰 검색 (심리/의도 파악용)
# ---------------------------------------------------------
def get_driver_interview(driver: str, event: str = "") -> str:
    """
    특정 드라이버나 관계자의 인터뷰, 발언, 심정을 검색합니다.
    (예: "Verstappen", "Monaco GP")
    """
    if not retriever_engine: return "⚠️ 검색 엔진 오류"
    
    # 💡 [Prompt Engineering] 검색어 뒤에 'interview', 'quotes' 등을 붙여 인터뷰 기사 유도
    query = f"{driver} {event} interview quotes reaction said statement"
    print(f" [Search] Interview: '{query}'")
    
    # 인터뷰는 정확도가 중요하므로 threshold를 약간 높게(0.5)
    results = retriever_engine.search(query, limit=4, score_threshold=0.5)
    
    if not results:
        return f"'{driver}' 선수의 관련 인터뷰를 찾지 못했습니다."
        
    return f"##  {driver} 인터뷰/발언 검색 결과:\n" + _format_rag_results(results)


# ---------------------------------------------------------
#  2. 기술/업데이트 분석 (차량 성능 파악용)
# ---------------------------------------------------------
def search_technical_analysis(team: str, component: str = "") -> str:
    """
    팀의 기술 업데이트, 차량 문제, 공기역학 분석 리포트를 검색합니다.
    (예: "Ferrari", "Floor upgrade")
    """
    if not retriever_engine: return " 검색 엔진 오류"
    
    # 기술 용어 가중치 추가
    query = f"{team} {component} technical analysis upgrade aerodynamics performance issues"
    print(f" [Search] Tech: '{query}'")
    
    results = retriever_engine.search(query, limit=3, score_threshold=0.55)
    
    return f"##  {team} 기술 분석 리포트:\n" + _format_rag_results(results)


# ---------------------------------------------------------
#  3. 규정 및 판례 검색 (전략/시뮬레이션용)
# ---------------------------------------------------------
def search_regulation_precedent(keyword: str) -> str:
    """
    FIA 규정 위반, 페널티 사례, 심판 판정 등을 검색합니다.
    (예: "impeding penalty", "track limits")
    """
    if not retriever_engine: return " 검색 엔진 오류"
    
    query = f"{keyword} FIA steward decision penalty regulation rule breach"
    print(f" [Search] Regulation: '{query}'")
    
    results = retriever_engine.search(query, limit=3, score_threshold=0.5)
    
    return f"##  규정 및 페널티 사례:\n" + _format_rag_results(results)


# ---------------------------------------------------------
#  4. 타임라인/일반 뉴스 (브리핑용)
# ---------------------------------------------------------
def get_event_timeline(topic: str) -> str:
    """
    특정 주제나 그랑프리의 전반적인 흐름(Timeline)을 파악합니다.
    """
    if not retriever_engine: return "⚠️ 검색 엔진 오류"
    
    print(f" [Search] Timeline: '{topic}'")
    results = retriever_engine.search(topic, limit=5, score_threshold=0.5)
    
    return f"##  '{topic}' 관련 뉴스 요약:\n" + _format_rag_results(results)


# ---------------------------------------------------------
#  5. Web 검색 (최신 정보 보완 - DuckDuckGo)
# ---------------------------------------------------------
def search_f1_news_web(query: str) -> str:
    """
    (Legacy) RAG에 없는 최신 실시간 정보를 웹에서 검색합니다.
    """
    print(f" [Web Search] '{query}'")
    try:
        results = []
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=3))
            for r in ddg_results:
                results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSummary: {r.get('body')}")
        return "\n---\n".join(results) if results else "검색 결과 없음"
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"웹 검색 오류: {e}"


# --- 테스트 실행부 (Main) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚦 PitWall RAG Tools Test")
    print("="*50)

    # 1. 기술 분석 테스트
    print(search_technical_analysis("Mercedes", "update"))
    
    print("\n" + "-"*30 + "\n")
    
    # 2. 인터뷰 검색 테스트
    print(get_driver_interview("Verstappen", "retirement"))

    print("\n" + "-"*30 + "\n")

    # 3. 규정 관련 테스트
    print(search_regulation_precedent("two move"))