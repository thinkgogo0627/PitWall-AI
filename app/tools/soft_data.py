# app/tools/soft_data.py
#
# RAG 검색 도구 — 통합 버전
#
# 변경사항:
#   - DuckDuckGo (search_f1_news_web) 완전 제거
#   - 목적별 4개 함수 → search_f1_context() 단일 함수로 통합
#   - retriever.py의 F1Retriever 직접 활용

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_pipeline.retriever import F1Retriever

logger = logging.getLogger(__name__)

# ────────────────────────────────────────
# 1. RAG 엔진 초기화 (Global Instance)
# ────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

try:
    retriever_engine = F1Retriever(
        qdrant_url=QDRANT_URL,
        collection_name="f1_knowledge_base"
    )
    print("✅ RAG Search Engine Ready.")
except Exception as e:
    print(f"❌ RAG Engine Load Failed: {e}")
    retriever_engine = None


# ────────────────────────────────────────
# 2. Helper: 검색 결과 → LLM 프롬프트용 텍스트 포맷팅
# ────────────────────────────────────────
def _format_rag_results(results: list, max_chars: int = 2000) -> str:
    """
    retriever.search()가 반환한 dict 리스트를 LLM이 읽기 좋은 형태로 변환.
    각 문서는 제목/출처/날짜/유사도/내용 순서로 정리.
    """
    if not results:
        return "관련 정보를 찾지 못했습니다."

    context_list = []
    for i, hit in enumerate(results, 1):
        score   = hit.get('score', 0.0)
        title   = hit.get('title', 'No Title')
        source  = hit.get('platform', 'Unknown')
        date    = str(hit.get('published_at', ''))[:10]
        text    = hit.get('text', '').strip()

        # 너무 긴 본문은 잘라서 토큰 절약
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        context_list.append(
            f"[{i}] 제목: {title}\n"
            f"    출처: {source} ({date}) | 유사도: {score:.3f}\n"
            f"    내용: {text}"
        )

    return "\n\n".join(context_list)


# ────────────────────────────────────────
# 3. 핵심 함수: 통합 RAG 검색
# ────────────────────────────────────────
def search_f1_context(query: str, limit: int = 4, score_threshold: float = 0.45) -> str:
    """
    [통합 RAG 검색]
    사용자 질문을 그대로 받아서 Qdrant에서 가장 관련성 높은 문서를 검색합니다.
    브리핑 에이전트의 generate_quick_summary에서 Data Injection 용도로 사용.

    Args:
        query           : 검색 쿼리 (사용자 입력 또는 year+gp 조합)
        limit           : 반환할 최대 문서 수 (기본 4개)
        score_threshold : 최소 유사도 점수 (기본 0.45)

    Returns:
        LLM 프롬프트에 주입할 포맷팅된 문자열
    """
    if not retriever_engine:
        return "[RAG_UNAVAILABLE] RAG 엔진을 사용할 수 없습니다."

    print(f"🔍 [RAG Search] '{query}' (limit={limit}, threshold={score_threshold})")

    try:
        results = retriever_engine.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )

        if not results:
            return f"[RAG_NO_RESULT] '{query}'에 관련된 문서를 찾지 못했습니다."

        print(f"   → {len(results)}개 문서 검색됨 (최고 유사도: {results[0].get('score', 0):.3f})")
        return _format_rag_results(results)

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return f"[RAG_ERROR] 검색 중 오류가 발생했습니다: {e}"


# ────────────────────────────────────────
# 4. (하위 호환) 기존 함수명 유지 — briefing_agent import 오류 방지
#    나중에 briefing_agent.py 정리 후 삭제 예정
# ────────────────────────────────────────
def get_driver_interview(driver: str, event: str = "") -> str:
    query = f"{driver} {event} interview quotes reaction"
    return search_f1_context(query, limit=4)

def search_technical_analysis(team: str, component: str = "") -> str:
    query = f"{team} {component} technical analysis upgrade aerodynamics"
    return search_f1_context(query, limit=3)

def search_regulation_precedent(keyword: str) -> str:
    query = f"{keyword} FIA regulation penalty rule"
    return search_f1_context(query, limit=3)

def get_event_timeline(topic: str) -> str:
    return search_f1_context(topic, limit=5)


# ────────────────────────────────────────
# 테스트 실행
# ────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Ferrari Macarena rear wing 2026",
        "Mercedes power unit compression ratio controversy",
        "Piastri DNS Chinese GP reason",
        "Russell 2026 Australian GP win strategy",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"쿼리: {q}")
        print('='*60)
        result = search_f1_context(q, limit=3)
        print(result)