import os
from llama_index.core.tools import FunctionTool
from data_pipeline.retriever import F1Retriever

## 규정집만 찝어서 검색하도록 메타데이터 필터를 거는 역할 수행


retriever = F1Retriever(collection_name="f1_news")

def search_fia_regulations(query: str) -> str:
    """
    [RAG] FIA 공식 규정집(Technical/Sporting Regulations)을 검색합니다.
    """
    # 2. 우리의 Retriever 사용 (필터링 적용!)
    results = retriever.search(
        query=query, 
        limit=4, 
        filter_meta={"platform": "FIA Official PDF"} # 👈 규정집만 쏙 골라냄
    )
    
    if not results:
        return "관련된 규정 조항을 찾을 수 없습니다."

    # 3. 검색 결과를 에이전트가 읽기 좋은 문자열로 변환
    formatted_response = ""
    for idx, item in enumerate(results, 1):
        title = item.get('title', 'Untitled')
        content = item.get('content', '')
        score = item.get('score', 0.0)
        formatted_response += f"\n[Document {idx} - {title} (Sim: {score:.2f})]\n{content}\n"
        
    return formatted_response

# 4. 도구 포장
regulation_tool = FunctionTool.from_defaults(
    fn=search_fia_regulations,
    name="Search_FIA_Regulations",
    description="2025/2026 F1 기술 및 스포팅 규정(PDF)을 검색합니다. 조항(Article) 기반의 팩트 체크 시 사용하세요."
)