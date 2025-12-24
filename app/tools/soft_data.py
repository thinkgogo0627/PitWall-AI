## RAG 검색 도구 구현

## 데이터 파이프라인을 통해 크롤링되어 ChromaDB에 저장된 벡터 검색 기능 구현


import sys
import os
import logging
from duckduckgo_search import DDGS


# 프로젝트 루트 경로 추가 (모듈 import용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from database.vector_store import F1VectorStore
from llama_index.core import VectorStoreIndex


# --- 1. 검색 엔진 초기화 (전역 변수로 한 번만 로드) ---
print(" Soft Data(ChromaDB) 연결 및 검색 엔진 로딩 중...")

try:
    # DB 인스턴스 생성 (이미 저장된 데이터에 연결됨)
    f1_store = F1VectorStore()
    
    # 저장된 Vector Store로부터 Index를 다시 로드 (Re-indexing 아님, 로드임)
    # embed_model은 F1VectorStore가 초기화될 때 Settings에 설정된 것을 사용
    index = VectorStoreIndex.from_vector_store(
        vector_store=f1_store.vector_store,
    )
    
    # Retriever 생성 (상위 3~5개 문서만 가져오도록 설정)
    # similarity_top_k=5 -> 상위 5개 문서만 가져오기
    retriever_engine = index.as_retriever(similarity_top_k=5)
    
    print(" 검색 엔진 준비 완료!")

except Exception as e:
    print(f" 검색 엔진 로드 실패: {e}")
    retriever_engine = None


# --- 2. 검색 도구 함수 정의 (LLM이 사용할 함수) ---
def search_f1_news(query: str):
    """
    F1 뉴스, 드라이버 인터뷰, 기술 분석 리포트, 규정집 등을 검색하여 텍스트로 반환합니다.
    경기 전략의 배경, 사고 원인, 팀의 공식 입장 등을 파악할 때 사용.
    
    Args:
        query (str): 검색할 질문 키워드 (예: "페라리 전략 실패 원인", "베르스타펜 인터뷰")
    """
    if not retriever_engine:
        return "검색 엔진이 초기화되지 않아 정보를 찾을 수 없습니다."
        
    print(f" RAG 검색 수행: '{query}'")
    
    # 검색 실행
    results = retriever_engine.retrieve(query)
    
    if not results:
        return "관련된 뉴스나 리포트를 찾지 못했습니다."
    
    # 검색 결과를 하나의 문자열로 예쁘게 포장
    context_list = []
    for i, node in enumerate(results, 1):
        # 메타데이터에서 제목과 출처 추출
        title = node.metadata.get('title', '제목 없음')
        source = node.metadata.get('source', '알 수 없음')
        date = node.metadata.get('crawled_at', '').split('T')[0] # 날짜만
        
        # 본문 내용 (너무 길면 자르거나 그대로 사용)
        text = node.text.strip()
        
        # 포맷팅
        context_list.append(f"[{i}] 제목: {title} (출처: {source}, {date})\n내용: {text}\n")
        
    final_context = "\n---\n".join(context_list)
    
    return final_context

# 로거 설정
logger = logging.getLogger(__name__)

def search_f1_news_web(query: str) -> str:
    """
    DuckDuckGo를 사용하여 실제 인터넷에서 F1 관련 뉴스/정보를 검색합니다.
    최신 뉴스, 인터뷰, 기술 분석 기사 등을 찾을 때 사용됩니다.
    """
    print(f" [Web Search] 검색어: '{query}'")
    
    try:
        results = []
        # ddg 객체 생성 (컨텍스트 매니저 사용 권장)
        with DDGS() as ddgs:
            # 검색 실행 (상위 5개 결과만)
            ddg_results = list(ddgs.text(query, max_results=5))
            
            for r in ddg_results:
                title = r.get('title', 'No Title')
                link = r.get('href', 'No Link')
                body = r.get('body', 'No Content')
                results.append(f"Title: {title}\nLink: {link}\nSummary: {body}\n")

        if not results:
            return "검색 결과가 없습니다."
            
        # 검색 결과를 하나의 텍스트로 합쳐서 반환
        return "\n---\n".join(results)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"인터넷 검색 중 오류 발생: {e}"



# --- 테스트 실행 (파일 직접 실행 시) ---
if __name__ == "__main__":
    # 테스트 질문
    test_query = "키미 안토넬리의 올해 레이스 실적"
    print(search_f1_news(test_query))