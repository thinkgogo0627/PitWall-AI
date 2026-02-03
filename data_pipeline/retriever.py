import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class F1Retriever:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "f1_knowledge_base"):
        """
        RAG 검색 엔진 초기화
        """
        self.collection_name = collection_name
        
        # 1. Qdrant 클라이언트 연결
        # (Docker 내부에서는 'http://qdrant:6333', 로컬에서는 'http://localhost:6333')
        print(f" [Retriever] Qdrant 연결 중... ({qdrant_url})")
        self.client = QdrantClient(url=qdrant_url)
        
        # 2. 임베딩 모델 로드
        docker_model_path = "/opt/airflow/data/model_cache/bge-m3"
        # 2. 로컬 테스트용 경로 (Docker 밖에서 돌릴 때)
        local_model_path = os.path.join(os.path.dirname(__file__), "../data/model_cache/bge-m3")
        
        if os.path.exists(docker_model_path):
            print(f" [Retriever] 로컬 모델 로드 (Docker): {docker_model_path}")
            model_source = docker_model_path
        
        elif os.path.exists(local_model_path):
            print(f" [Retriever] 로컬 모델 로드 (Local): {local_model_path}")
            model_source = local_model_path
        
        else:
            print(" [Retriever] 로컬 모델 없음! HuggingFace에서 다운로드합니다.")
            model_source = 'BAAI/bge-m3'

        self.embed_model = SentenceTransformer(model_source)
        print(" [Retriever] 준비 완료.")
        
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.4, filter_meta: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        자연어 질문을 벡터로 변환하여 Qdrant에서 유사한 문서를 찾습니다.
        """
        try:
            # 1. 질문을 벡터로 변환 (Encoding)
            query_vector = self.embed_model.encode(query).tolist()
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

            # 2. 벡터 검색 (Search)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector, # 벡터 전달
                query_filter = query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold # 유사도 커트라인
            ).points
            
            # 3. 결과 정리 (Payload 추출)
            results = []
            for hit in search_result:
                payload = hit.payload
                payload['score'] = hit.score # 유사도 점수 추가
                results.append(payload)
                
            return results

        except Exception as e:
            print(f"검색 중 에러 발생: {e}")
            return []

if __name__ == "__main__":
    # 간단 테스트
    retriever = F1Retriever()
    print(retriever.search("Red Bull update"))