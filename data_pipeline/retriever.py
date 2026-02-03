import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models  # 👈 필터링 모델 추가
from sentence_transformers import SentenceTransformer

class F1Retriever:
    def __init__(self, qdrant_url: str = None, collection_name: str = "f1_knowledge_base"):
        # ... (기존 초기화 코드 100% 동일) ...
        if not qdrant_url:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # 모델 로드 로직 (기존 유지)
        model_source = 'BAAI/bge-m3' # 혹은 로컬 경로
        self.embed_model = SentenceTransformer(model_source)

    # 👇 [수정] filter_meta 인자 추가!
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.4, filter_meta: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        [Upgrade] 메타데이터 필터링 지원
        filter_meta 예시: {"platform": "FIA Official PDF"}
        """
        try:
            # 1. 인코딩
            query_vector = self.embed_model.encode(query).tolist()
            
            # 2. 필터 객체 생성 (Qdrant 방식)
            query_filter = None
            if filter_meta:
                must_conditions = []
                for key, value in filter_meta.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                query_filter = models.Filter(must=must_conditions)

            # 3. 검색 (query_filter 적용)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,  # 👈 여기에 필터 꽂기
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold
            ).points
            
            # 4. 결과 정리
            results = []
            for hit in search_result:
                payload = hit.payload
                payload['score'] = hit.score
                results.append(payload)
                
            return results

        except Exception as e:
            print(f"검색 중 에러 발생: {e}")
            return []