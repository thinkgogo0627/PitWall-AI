from llama_index.core.tools import FunctionTool
# 👇 우리가 만든 클래스 임포트 (경로는 실제 파일 위치에 맞게!)
from app.modules.retriever import F1Retriever 

# 1. 리트리버 인스턴스 생성 (싱글톤)
# 규정집이 'f1_news' 컬렉션에 함께 들어있다고 가정 (Crawler에서 그렇게 넣었으므로)
retriever = F1Retriever(collection_name="f1_news") 

def search_fia_regulations(query: str) -> str:
    """
    [RAG] FIA 공식 규정집(Technical/Sporting Regulations)을 검색합니다.
    """
    # 2. 커스텀 리트리버 사용 + 필터링 적용
    results = retriever.search(
        query=query, 
        limit=4, 
        # 👇 Crawler에서 저장할 때 썼던 그 메타데이터 키값!
        filter_meta={"platform": "FIA Official PDF"} 
    )
    
    if not results:
        return "관련된 규정 조항을 찾을 수 없습니다."

    # 3. 에이전트가 읽기 좋게 포맷팅
    formatted_response = ""
    for idx, item in enumerate(results, 1):
        title = item.get('title', 'Untitled')
        content = item.get('content', '')
        # page_no가 있다면 표시
        page = item.get('page_no', '?')
        
        formatted_response += f"\n[Document {idx} | {title} (Page {page})]\n{content}\n"
        
    return formatted_response

# 4. LlamaIndex 도구로 포장
regulation_tool = FunctionTool.from_defaults(
    fn=search_fia_regulations,
    name="Search_FIA_Regulations",
    description="2025/2026 F1 기술 및 스포팅 규정(PDF)을 검색합니다. 조항(Article) 기반의 팩트 체크 시 사용하세요."
)