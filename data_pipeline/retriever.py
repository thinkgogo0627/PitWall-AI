import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

class F1Retriever:
    def __init__(self, qdrant_url: str = None, collection_name: str = "f1_knowledge_base"):
        if not qdrant_url:
            qdrant_url = os.getenv("QDRANT_URL")
        
        self.client = QdrantClient(url=qdrant_url,
                                   api_key= os.getenv('QDRANT_API_KEY'))
        self.collection_name = collection_name
        
        # [수정] 무거운 로컬 모델 대신 구글 API 모델 로드
        # model_source = 'BAAI/bge-m3' (삭제)
        # self.embed_model = SentenceTransformer(model_source) (삭제)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        print("🔌 [Retriever] Loading Google Gemini Embedding Model...")
        
        self.embed_model = GoogleGenAIEmbedding(
            model_name="models/gemini-embedding-001", 
            api_key=api_key
        )

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.4, filter_meta: Optional[Dict] = None) -> List[Dict[str, Any]]:
        try:
            # 1. 인코딩 [핵심 수정 포인트!]
            # 기존: query_vector = self.embed_model.encode(query).tolist()
            # 변경: API 방식 메서드 사용
            query_vector = self.embed_model.get_query_embedding(query)
            
            # 2. 필터 객체 생성 (기존 동일)
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

            # 3. 검색 (기존 동일)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold
            ).points
            
            # 4. 결과 정리 (기존 동일)
            results = []
            for hit in search_result:
                payload = hit.payload
                payload['score'] = hit.score
                results.append(payload)
                
            return results

        except Exception as e:
            print(f"검색 중 에러 발생: {e}")
            return []