import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

## 규정집만 찝어서 검색하도록 메타데이터 필터를 거는 역할 수행

# Qdrant 연결 설정 (환경 변수 사용)
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def search_fia_regulations(query: str) -> str:
    """
    [RAG] FIA 공식 규정집(Technical/Sporting Regulations)을 검색합니다.
    사용자의 질문과 관련된 조항(Article)을 찾아 원문을 반환합니다.
    """
    try:
        # 1. Qdrant 연결
        vector_store = QdrantVectorStore(client=client, collection_name="f1_news") # 뉴스랑 같은 컬렉션 쓴다고 가정
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 2. 필터링: 'platform'이 'FIA Official PDF'인 문서만 타겟팅
        # (Crawler 코드에서 platform="FIA Official PDF"로 저장했으므로)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="platform", value="FIA Official PDF")
            ]
        )

        # 3. 검색 엔진 생성
        query_engine = index.as_query_engine(
            similarity_top_k=5, # 관련 조항 5개 참조
            filters=filters
        )

        # 4. 검색 수행
        response = query_engine.query(query)
        return str(response)

    except Exception as e:
        return f"규정 검색 중 오류 발생: {e}"

# 도구 포장
regulation_tool = FunctionTool.from_defaults(
    fn=search_fia_regulations,
    name="FIA_Regulation_Search",
    description="2025/2026 F1 기술 및 스포팅 규정(PDF)을 검색합니다. 규정 위반, 페널티, 기술 변경 사항 질문에 사용하세요."
)