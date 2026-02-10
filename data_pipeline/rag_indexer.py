import os
import uuid
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import torch # GPU 체크용

# 도메인 모델 (Beanie Document)
from domain.documents import F1NewsDocument

class RAGIndexer:
    def __init__(self, mongo_uri: str, qdrant_url: str):
        self.mongo_uri = mongo_uri
        self.qdrant_url = qdrant_url
        self.collection_name = "f1_knowledge_base"
        
        # 1. Qdrant 클라이언트 연결
        print(f"🔌 [Indexer] Connecting to Qdrant: {self.qdrant_url}")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        
        # 2. 임베딩 모델 로드 (GPU 가속 확인)
        # 로컬 캐시 경로 우선 확인 (로컬에 설정한 그 경로!)
        docker_model_path = "/app/models/bge-m3"
        local_model_path = os.path.join(os.path.dirname(__file__), "../data/model_cache/bge-m3")
        
        model_path = "BAAI/bge-m3" # 기본값
        if os.path.exists(docker_model_path):
            print(f"📂 Loading from docker Path: {docker_model_path}")
            model_path = docker_model_path
        elif os.path.exists(local_model_path):
            print(f"💻 Loading from Local Path: {local_model_path}")
            model_path = local_model_path
        else:
            print("🌐 Model not found locally. Downloading from HuggingFace Hub...")

        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" [Indexer] Loading Model on [{device.upper()}] from {model_path}...")
        
        self.embed_model = SentenceTransformer(model_path, device=device)

    def _generate_deterministic_uuid(self, text: str) -> str:
        """URL 기반으로 항상 같은 UUID를 생성 (멱등성 보장 핵심)"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

    async def run_indexing(self, batch_size: int = 50):
        """MongoDB의 데이터를 읽어서 Qdrant에 적재"""
        print(" Indexing Started...")
        
        # 1. MongoDB 연결
        client = AsyncIOMotorClient(self.mongo_uri)
        db = client.pitwall_db
        # Beanie 초기화가 안 되어 있을 수 있으므로 raw query 사용
        # (Beanie 의존성을 줄여서 가볍게 실행)
        collection = db.get_collection("f1_news_articles")
        
        # 2. Qdrant 컬렉션 생성 (없으면 생성) & 인덱스 설정
        if not self.qdrant_client.collection_exists(self.collection_name):
            print(f" Creating Collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024, # bge-m3 output dimension
                    distance=models.Distance.COSINE
                )
            )
            # [최적화] 필터링 자주 하는 필드 인덱싱
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="platform",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="published_at",
                field_schema=models.PayloadSchemaType.DATETIME
            )

        # 3. 데이터 패치 및 임베딩 (Batch Processing)
        # 아직 벡터화되지 않은(혹은 전체) 문서를 가져옵니다.
        # 여기서는 간단하게 '전체 스캔 후 Upsert' 방식으로 멱등성을 테스트합니다.
        cursor = collection.find({}) 
        
        batch_docs = []
        total_indexed = 0
        
        async for doc in cursor:
            batch_docs.append(doc)
            
            if len(batch_docs) >= batch_size:
                await self._process_batch(batch_docs)
                total_indexed += len(batch_docs)
                batch_docs = [] # 초기화

        # 남은 배치 처리
        if batch_docs:
            await self._process_batch(batch_docs)
            total_indexed += len(batch_docs)
            
        print(f" Indexing Finished! Total {total_indexed} documents processed.")

    async def _process_batch(self, docs: List[dict]):
        """배치 단위 임베딩 및 업로드"""
        texts = [d.get('content', '')[:8000] for d in docs] # 너무 긴 텍스트 잘라내기
        metadatas = []
        ids = []
        
        for d in docs:
            url = d.get('url', '')
            # [멱등성 핵심] URL로 ID 생성
            ids.append(self._generate_deterministic_uuid(url))
            
            metadatas.append({
                "title": d.get('title', ''),
                "url": url,
                "platform": d.get('platform', 'Unknown'),
                "published_at": d.get('published_at', '').isoformat() if d.get('published_at') else None,
                "text": d.get('content', '')[:1000] # Qdrant Payload에 저장할 본문 (검색 결과 표시용)
            })

        # 1. GPU로 한 방에 임베딩
        if not texts: return
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        
        # 2. Qdrant Points 생성
        points = [
            models.PointStruct(
                id=id_,
                vector=embedding.tolist(),
                payload=metadata
            )
            for id_, embedding, metadata in zip(ids, embeddings, metadatas)
        ]
        
        # 3. 업로드 (Upsert: 기존 ID 있으면 덮어씀)
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f" Batch Upserted: {len(points)} items")

# 테스트 실행용
if __name__ == "__main__":
    import asyncio
    indexer = RAGIndexer(
        mongo_uri="mongodb://localhost:27017", # 로컬 테스트 시
        qdrant_url="http://localhost:6333"
    )
    asyncio.run(indexer.run_indexing())