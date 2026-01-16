# data/soft_data.py

## RAG 검색 도구 구현
## 

import sys
import os
import logging
from duckduckgo_search import DDGS

# [경로 설정]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# [NEW] Qdrant 검색기 임포트
from data_pipeline.retriever import F1Retriever

# 로거 설정
logger = logging.getLogger(__name__)

# --- 1. 검색 엔진 초기화 ---
# Docker 환경변수 지원
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

print(f"🔌 Qdrant 연결 시도 중... ({QDRANT_URL})")

try:
    retriever_engine = F1Retriever(qdrant_url=QDRANT_URL)
    print("✅ PitWall RAG 엔진 시동 완료!")
except Exception as e:
    print(f"❌ 검색 엔진 로드 실패: {e}")
    retriever_engine = None

# ---------------------------------------------------------
# 🛠️ Helper: 검색 결과 포맷팅 함수
# ---------------------------------------------------------
def _format_rag_results(results: list) -> str:
    if not results:
        return "관련 정보를 찾지 못했습니다."
    
    context_list = []
    for i, hit in enumerate(results, 1):
        score = hit.get('score', 0.0)
        title = hit.get('title', 'No Title')
        source = hit.get('source', 'Unknown')
        date = hit.get('published_at', '')[:10]
        text = hit.get('text', '').strip()
        
        # 가독성을 위해 본문 길이 조정 (선택 사항)
        if len(text) > 500:
            text = text[:500] + "...(more)"

        context_list.append(
            f"[{i}] {title} (Source: {source} | Date: {date} | Score: {score:.2f})\n"
            f"    \"{text}\""
        )
    return "\n\n".join(context_list)

# ---------------------------------------------------------
# 🧠 1. 드라이버 인터뷰 검색 (Briefing/Strategy Agent용)
# ---------------------------------------------------------
def get_driver_interview(driver: str, event: str = "") -> str:
    """
    드라이버의 인터뷰, 코멘트, 심정을 검색합니다.
    Args:
        driver: 드라이버 이름 (예: "Verstappen", "Hamilton")
        event: 관련 이벤트 (예: "Monaco GP Qualifying")
    """
    if not retriever_engine: return "검색 엔진 오류"
    
    # 검색어 확장 (Query Expansion)
    query = f"{driver} {event} interview quotes reaction said"
    print(f"🎤 [Interview Search] Query: '{query}'")
    
    results = retriever_engine.search(query, limit=4, score_threshold=0.5)
    
    if not results:
        return f"{driver} 선수의 관련 인터뷰를 찾을 수 없습니다."
        
    return f"## {driver} 인터뷰 검색 결과:\n" + _format_rag_results(results)

# ---------------------------------------------------------
# 🔧 2. 기술 업데이트 분석 (Circuit/Simulation Agent용)
# ---------------------------------------------------------
def search_technical_analysis(team: str, component: str = "") -> str:
    """
    특정 팀의 차량 업데이트나 기술적인 문제를 검색합니다.
    Args:
        team: 팀 이름 (예: "Ferrari", "Red Bull")
        component: 부품명 (예: "Floor", "Sidepod", "Engine")
    """
    if not retriever_engine: return "검색 엔진 오류"
    
    query = f"{team} {component} technical analysis upgrade update aerodynamics problem"
    print(f"🛠️ [Tech Search] Query: '{query}'")
    
    # 기술 분석은 Autosport 소스가 더 정확하므로 필터링(가능하다면)하면 좋지만, 
    # 일단 검색어로 가중치를 줌.
    results = retriever_engine.search(query, limit=3, score_threshold=0.6)
    
    return f"## {team} 기술 분석 리포트:\n" + _format_rag_results(results)

# ---------------------------------------------------------
# 📜 3. 규정 및 페널티 사례 (Strategy Agent용)
# ---------------------------------------------------------
def search_regulation_precedent(incident_type: str) -> str:
    """
    특정 사건에 대한 FIA 규정이나 과거 페널티 사례를 검색합니다.
    Args:
        incident_type: 사건 유형 (예: "impeding in qualifying", "pit lane speeding")
    """
    if not retriever_engine: return "검색 엔진 오류"
    
    query = f"{incident_type} penalty FIA stewards decision regulation precedent"
    print(f"⚖️ [Regulation Search] Query: '{query}'")
    
    results = retriever_engine.search(query, limit=3, score_threshold=0.55)
    
    return f"## 규정 및 페널티 사례 검색:\n" + _format_rag_results(results)

# ---------------------------------------------------------
# 📰 4. 타임라인/일반 뉴스 (Briefing Agent용)
# ---------------------------------------------------------
def get_event_timeline(grand_prix: str) -> str:
    """
    특정 그랑프리 주간의 주요 사건을 검색합니다.
    """
    if not retriever_engine: return "검색 엔진 오류"
    
    query = f"{grand_prix} weekend summary highlights timeline key moments"
    print(f"📅 [Timeline Search] Query: '{query}'")
    
    results = retriever_engine.search(query, limit=5, score_threshold=0.5)
    
    return f"## {grand_prix} 주요 타임라인:\n" + _format_rag_results(results)


# --- 5. Web 검색 도구 (DuckDuckGo - 최신 정보 보완용) ---
def search_f1_news_web(query: str) -> str:
    """실시간 웹 검색 (기존 유지)"""
    print(f"🌐 [Web Search] Query: '{query}'")
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

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 데이터가 266개나 있으니 뭐라도 나와야 합니다!
    print(get_driver_interview("Verstappen", "Qualifying"))
    print("\n" + "="*50 + "\n")
    print(search_technical_analysis("Ferrari", "updates"))